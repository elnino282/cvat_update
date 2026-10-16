# Copyright (C) CVAT.ai Corporation
# SPDX-License-Identifier: MIT

"""Back up, migrate, and roll back 3D Car cuboid annotations safely."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.core.serializers.json import DjangoJSONEncoder
from django.db import transaction

from cvat.apps.engine import models


VEHICLE_TYPES = ["car", "truck", "bus", "van", "other"]
def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), cls=DjangoJSONEncoder)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _verify_backup_integrity(data):
    if (not isinstance(data, dict) or not isinstance(data.get("before"), dict) or
            _digest(data["before"]) != data.get("beforeSHA256")):
        raise CommandError("Backup checksum mismatch; migration stopped without changes")


def _schema(label):
    if label is None:
        return None
    return {
        "id": label.id, "name": label.name, "type": label.type, "color": label.color,
        "task_id": label.task_id, "project_id": label.project_id,
        "attributes": list(label.attributespec_set.order_by("id").values(
            "id", "name", "mutable", "input_type", "default_value", "values",
        )),
    }


def _rows(model, queryset):
    return list(queryset.order_by("pk").values(*[field.attname for field in model._meta.concrete_fields]))


def _capture(scope, source, target):
    label_ids = [item for item in (source.id if source else None, target.id if target else None) if item]
    base = {"job__segment__task_id__in": scope["guard_task_ids"], "label_id__in": label_ids}
    shapes = models.LabeledShape.objects.filter(**base)
    tracks = models.LabeledTrack.objects.filter(**base)
    images = models.LabeledImage.objects.filter(**base)
    intervals = models.LabeledInterval.objects.filter(**base)
    track_ids = list(tracks.values_list("id", flat=True))
    shape_ids = list(shapes.values_list("id", flat=True))
    tracked_shapes = models.TrackedShape.objects.filter(track_id__in=track_ids)
    tracked_shape_ids = list(tracked_shapes.values_list("id", flat=True))
    return {
        "labels": [item for item in (_schema(source), _schema(target)) if item],
        "shapes": _rows(models.LabeledShape, shapes),
        "tracks": _rows(models.LabeledTrack, tracks),
        "tracked_shapes": _rows(models.TrackedShape, tracked_shapes),
        "shape_attributes": _rows(models.LabeledShapeAttributeVal,
                                   models.LabeledShapeAttributeVal.objects.filter(shape_id__in=shape_ids)),
        "track_attributes": _rows(models.LabeledTrackAttributeVal,
                                  models.LabeledTrackAttributeVal.objects.filter(track_id__in=track_ids)),
        "tracked_shape_attributes": _rows(models.TrackedShapeAttributeVal,
                                           models.TrackedShapeAttributeVal.objects.filter(shape_id__in=tracked_shape_ids)),
        "images": _rows(models.LabeledImage, images),
        "intervals": _rows(models.LabeledInterval, intervals),
    }


def _model_rows(state):
    return (
        (models.LabeledShapeAttributeVal, state["shape_attributes"]),
        (models.LabeledTrackAttributeVal, state["track_attributes"]),
        (models.TrackedShapeAttributeVal, state["tracked_shape_attributes"]),
        (models.TrackedShape, state["tracked_shapes"]),
        (models.LabeledShape, state["shapes"]),
        (models.LabeledTrack, state["tracks"]),
    )


def _restore_rows(state, current_state):
    before_rows = dict(_model_rows(state))
    current_rows = dict(_model_rows(current_state))
    annotation_models = (models.LabeledShape, models.LabeledTrack, models.TrackedShape)
    attribute_models = (models.LabeledShapeAttributeVal, models.LabeledTrackAttributeVal,
                        models.TrackedShapeAttributeVal)
    for model in annotation_models:
        for row in before_rows[model]:
            values = {key: value for key, value in row.items() if key != "id"}
            if not model.objects.filter(pk=row["id"]).update(**values):
                raise CommandError(f"Annotation {model.__name__}:{row['id']} disappeared during rollback")
    for model in attribute_models:
        before = before_rows[model]
        before_ids = {row["id"] for row in before}
        current_ids = [row["id"] for row in current_rows[model]]
        model.objects.filter(pk__in=[row_id for row_id in current_ids if row_id not in before_ids]).delete()
        for row in before:
            values = {key: value for key, value in row.items() if key != "id"}
            model.objects.update_or_create(pk=row["id"], defaults=values)


def _write_backup(path, data, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (_canonical(data) + "\n").encode()
    if exclusive:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
        return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _touch_scope(scope):
    for job in models.Job.objects.filter(segment__task_id__in=scope["guard_task_ids"]):
        job.touch()
    tasks = list(models.Task.objects.filter(pk__in=scope["guard_task_ids"]).select_related("project"))
    for task in tasks:
        task.touch()
    project_ids = {task.project_id for task in tasks if task.project_id}
    for project in models.Project.objects.filter(pk__in=project_ids):
        project.touch()


class Command(BaseCommand):
    help = "Dry-run, back up, migrate, or roll back 3D Car cuboids to Vehicles/type=car."

    def add_arguments(self, parser):
        parser.add_argument("--task-id", type=int)
        parser.add_argument("--project-id", type=int)
        parser.add_argument("--mode", choices=("dry-run", "backup", "apply", "rollback"), default="dry-run")
        parser.add_argument("--backup", type=Path)
        parser.add_argument("--confirm-scope", help="Must equal task:<id> or project:<id> for apply.")

    def _scope(self, options):
        if bool(options["task_id"]) == bool(options["project_id"]):
            raise CommandError("Specify exactly one of --task-id or --project-id")
        if options["project_id"]:
            project = models.Project.objects.get(pk=options["project_id"])
            tasks = list(models.Task.objects.filter(project=project).order_by("id"))
            label_filter = {"project": project, "task__isnull": True, "parent__isnull": True}
            identity = f"project:{project.id}"
            guard_task_ids = [task.id for task in tasks]
        else:
            task = models.Task.objects.select_related("project").get(pk=options["task_id"])
            tasks = [task]
            if task.project_id:
                label_filter = {"project_id": task.project_id, "task__isnull": True, "parent__isnull": True}
                guard_task_ids = list(models.Task.objects.filter(project_id=task.project_id)
                                      .order_by("id").values_list("id", flat=True))
            else:
                label_filter = {"task": task, "project__isnull": True, "parent__isnull": True}
                guard_task_ids = [task.id]
            identity = f"task:{task.id}"
        if not tasks or any(task.dimension != models.DimensionType.DIM_3D for task in tasks):
            raise CommandError("The selected scope must contain only 3D tasks")
        if models.Task.objects.filter(pk__in=guard_task_ids).exclude(
                dimension=models.DimensionType.DIM_3D).exists():
            raise CommandError("A shared project label also belongs to a non-3D task")
        labels = models.Label.objects.filter(**label_filter)
        if labels.filter(name__iexact="Car").count() > 1 or labels.filter(name__iexact="Vehicles").count() > 1:
            raise CommandError("Duplicate Car or Vehicles labels make this scope ambiguous")
        source = labels.filter(name__iexact="Car").first()
        target = labels.filter(name__iexact="Vehicles").first()
        if source is None:
            raise CommandError("No Car label exists in the selected label scope")
        if source.type != "cuboid":
            raise CommandError("The Car source label must be a cuboid label")
        if target and target.type != "cuboid":
            raise CommandError("The existing Vehicles label is not a cuboid label")
        return {"task_ids": [task.id for task in tasks], "guard_task_ids": guard_task_ids,
                "label_filter": label_filter,
                "identity": identity, "source": source, "target": target}

    def _validate_annotations(self, scope, source):
        labels = [source.id]
        base = {"job__segment__task_id__in": scope["task_ids"], "label_id__in": labels}
        shapes = models.LabeledShape.objects.filter(**base)
        if shapes.exclude(type=models.ShapeType.CUBOID).exists() or shapes.filter(parent__isnull=False).exists():
            raise CommandError("Car contains non-cuboid or nested shapes; migration stopped")
        tracks = models.LabeledTrack.objects.filter(**base)
        if tracks.filter(parent__isnull=False).exists():
            raise CommandError("Car contains nested tracks; migration stopped")
        if models.LabeledImage.objects.filter(**base).exists() or models.LabeledInterval.objects.filter(**base).exists():
            raise CommandError("Car has image or interval annotations outside this cuboid migration")
        if models.TrackedShape.objects.filter(track__in=tracks).exclude(type=models.ShapeType.CUBOID).exists():
            raise CommandError("A Car track contains non-cuboid keyframes; migration stopped")

    def _preflight(self, scope):
        source, target = scope["source"], scope["target"]
        self._validate_annotations(scope, source)
        target_by_name = {spec.name: spec for spec in target.attributespec_set.all()} if target else {}
        for source_spec in source.attributespec_set.all():
            target_spec = target_by_name.get(source_spec.name)
            if target_spec and any(getattr(source_spec, key) != getattr(target_spec, key) for key in
                                   ("mutable", "input_type", "default_value", "values")):
                raise CommandError(f"Attribute schema conflict for {source_spec.name!r}")
            if source_spec.name == "type" and (
                    source_spec.mutable or source_spec.input_type != "select" or
                    source_spec.values.splitlines() != VEHICLE_TYPES):
                raise CommandError("Existing Car/type conflicts with the Vehicles enum")
        if target:
            type_spec = target_by_name.get("type")
            if type_spec and (type_spec.mutable or type_spec.input_type != "select" or
                              type_spec.values.splitlines() != VEHICLE_TYPES):
                raise CommandError("Existing Vehicles/type does not match the required immutable select")
        return True

    def _apply(self, scope):
        source, target = scope["source"], scope["target"]
        if target is None:
            target = models.Label.objects.create(
                name="Vehicles", type=models.LabelType.CUBOID, color=source.color,
                task_id=source.task_id, project_id=source.project_id,
            )
            scope["target"] = target
        target_specs = {spec.name: spec for spec in target.attributespec_set.all()}
        type_spec = target_specs.get("type")
        if type_spec is None:
            type_spec = models.AttributeSpec.objects.create(
                label=target, name="type", mutable=False, input_type="select",
                default_value="other", values="\n".join(VEHICLE_TYPES),
            )
            target_specs["type"] = type_spec
        mapping = {}
        for old_spec in source.attributespec_set.all():
            new_spec = target_specs.get(old_spec.name)
            if new_spec is None:
                new_spec = models.AttributeSpec.objects.create(
                    label=target, name=old_spec.name, mutable=old_spec.mutable,
                    input_type=old_spec.input_type, default_value=old_spec.default_value,
                    values=old_spec.values,
                )
                target_specs[old_spec.name] = new_spec
            mapping[old_spec.id] = new_spec.id

        task_filter = {"job__segment__task_id__in": scope["task_ids"]}
        shape_query = models.LabeledShape.objects.filter(**task_filter, label=source)
        track_query = models.LabeledTrack.objects.filter(**task_filter, label=source)
        shape_ids = list(shape_query.values_list("id", flat=True))
        track_ids = list(track_query.values_list("id", flat=True))
        tracked_shape_ids = list(models.TrackedShape.objects.filter(track_id__in=track_ids)
                                 .values_list("id", flat=True))
        attr_queries = (
            models.LabeledShapeAttributeVal.objects.filter(shape_id__in=shape_ids),
            models.LabeledTrackAttributeVal.objects.filter(track_id__in=track_ids),
            models.TrackedShapeAttributeVal.objects.filter(shape_id__in=tracked_shape_ids),
        )
        for queryset in attr_queries:
            for old_id, new_id in mapping.items():
                queryset.filter(spec_id=old_id).update(spec_id=new_id)
        shape_query.update(label=target)
        track_query.update(label=target)

        # Immutable values live on shapes or on the track, never on each keyframe.
        for shape in models.LabeledShape.objects.filter(pk__in=shape_ids):
            models.LabeledShapeAttributeVal.objects.update_or_create(
                shape=shape, spec=type_spec, defaults={"job_id": shape.job_id, "value": "car"},
            )
        for track in models.LabeledTrack.objects.filter(pk__in=track_ids):
            models.LabeledTrackAttributeVal.objects.update_or_create(
                track=track, spec=type_spec, defaults={"job_id": track.job_id, "value": "car"},
            )

    def handle(self, *args, **options):
        mode = options["mode"]
        backup_path = options["backup"]
        if mode in {"backup", "apply", "rollback"} and not backup_path:
            raise CommandError(f"--backup is required for mode={mode}")
        scope = self._scope(options)
        self._preflight(scope)
        before = _capture(scope, scope["source"], scope["target"])
        before_hash = _digest(before)
        if mode == "dry-run":
            source_shape_count = sum(row["label_id"] == scope["source"].id for row in before["shapes"])
            source_track_count = sum(row["label_id"] == scope["source"].id for row in before["tracks"])
            self.stdout.write(json.dumps({"status": "dry-run", "scope": scope["identity"],
                                          "taskIds": scope["task_ids"],
                                          "carShapes": source_shape_count,
                                          "carTracks": source_track_count,
                                          "sourceSHA256": before_hash}, sort_keys=True))
            return
        if mode == "backup":
            data = {"format": "cvat-autofit-vehicles-v1", "status": "backed-up",
                    "scope": scope["identity"], "taskIds": scope["task_ids"],
                    "before": before, "beforeSHA256": before_hash}
            _write_backup(backup_path, data, exclusive=True)
            self.stdout.write(f"Backup written: {backup_path} ({before_hash})")
            return

        if mode == "apply":
            if options["confirm_scope"] != scope["identity"]:
                raise CommandError(f"Apply requires --confirm-scope {scope['identity']}")
            try:
                data = json.loads(backup_path.read_text())
            except (OSError, ValueError) as exc:
                raise CommandError(f"Cannot read migration backup: {exc}") from exc
            _verify_backup_integrity(data)
            if (data.get("format") != "cvat-autofit-vehicles-v1" or
                    data.get("scope") != scope["identity"] or
                    data.get("status") not in {"backed-up", "applying", "applied"}):
                raise CommandError("A matching, unmodified backup is required before apply")
            if data.get("status") in {"applying", "applied"} and data.get("afterSHA256") == before_hash:
                data["status"] = "applied"
                _write_backup(backup_path, data)
                self.stdout.write(f"Migration already applied to {scope['identity']}")
                return
            if data.get("beforeSHA256") != before_hash:
                raise CommandError("A matching, unmodified backup is required before apply")
            data["status"] = "applying"
            _write_backup(backup_path, data)
            try:
                with transaction.atomic():
                    list(models.Job.objects.select_for_update().filter(
                        segment__task_id__in=scope["guard_task_ids"],
                    ))
                    latest = _capture(scope, scope["source"], scope["target"])
                    if _digest(latest) != before_hash:
                        raise CommandError("Annotations changed after backup; migration stopped")
                    self._apply(scope)
                    _touch_scope(scope)
                    after = _capture(scope, scope["source"], scope["target"])
                    data["afterSHA256"] = _digest(after)
                    data["after"] = after
                    data["targetLabelId"] = scope["target"].id
                    # Persist the intended post-state before commit so an interrupted
                    # process can distinguish a committed migration from a rollback.
                    _write_backup(backup_path, {**data, "status": "applying"})
            except Exception:
                data["status"] = "backed-up"
                _write_backup(backup_path, data)
                raise
            data["status"] = "applied"
            _write_backup(backup_path, data)
            self.stdout.write(f"Migration applied to {scope['identity']}; backup: {backup_path}")
            return

        try:
            data = json.loads(backup_path.read_text())
        except (OSError, ValueError) as exc:
            raise CommandError(f"Cannot read migration backup: {exc}") from exc
        _verify_backup_integrity(data)
        if data.get("format") != "cvat-autofit-vehicles-v1" or \
                data.get("status") not in {"applied", "rolling-back"} or data.get("scope") != scope["identity"]:
            raise CommandError("Rollback requires the applied backup for this exact scope")
        target_id = data.get("targetLabelId")
        target = models.Label.objects.filter(pk=target_id).first()
        current = _capture(scope, scope["source"], target)
        current_hash = _digest(current)
        if data.get("status") == "rolling-back" and current_hash == data.get("beforeSHA256"):
            data["status"] = "rolled-back"
            _write_backup(backup_path, data)
            self.stdout.write(f"Migration already rolled back for {scope['identity']}")
            return
        if data.get("status") == "rolling-back" and current_hash != data.get("afterSHA256"):
            raise CommandError("Annotations or schema changed during interrupted rollback")
        if data.get("status") == "applied" and current_hash != data.get("afterSHA256"):
            raise CommandError("Annotations or schema changed after migration; rollback stopped")
        with transaction.atomic():
            list(models.Job.objects.select_for_update().filter(
                segment__task_id__in=scope["guard_task_ids"],
            ).values_list("id", flat=True))
            current = _capture(scope, scope["source"], target)
            if _digest(current) != data.get("afterSHA256"):
                raise CommandError("Annotations changed during rollback preflight")
            _restore_rows(data["before"], current)
            before_label_ids = {item["id"] for item in data["before"]["labels"]}
            before_specs = {spec["id"] for label in data["before"]["labels"]
                            for spec in label["attributes"]}
            current_label_ids = [item["id"] for item in current["labels"]]
            created_labels = set(current_label_ids) - before_label_ids
            current_specs = {spec["id"] for label in current["labels"] for spec in label["attributes"]}
            for spec_id in current_specs - before_specs:
                models.AttributeSpec.objects.filter(pk=spec_id).delete()
            for label_id in created_labels:
                models.Label.objects.filter(pk=label_id).delete()
            _touch_scope(scope)
            restored_source = models.Label.objects.get(pk=scope["source"].id)
            restored_target = models.Label.objects.filter(pk=target_id).first()
            restored = _capture(scope, restored_source, restored_target)
            if _digest(restored) != data.get("beforeSHA256"):
                raise CommandError("Rollback result does not match the original snapshot")
            data["rollbackSHA256"] = _digest(restored)
            data["status"] = "rolling-back"
            _write_backup(backup_path, data)
        data["status"] = "rolled-back"
        _write_backup(backup_path, data)
        self.stdout.write(f"Migration rolled back for {scope['identity']}")
