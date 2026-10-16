# Copyright (C) CVAT.ai Corporation
# SPDX-License-Identifier: MIT

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase

from cvat.apps.engine import models
from cvat.apps.engine.management.commands import migrate_autofit_vehicles as migration


class AutoFitVehicleMigrationTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        data = models.Data.objects.create(size=2, start_frame=0, stop_frame=1, deleted_frames=[])
        cls.task = models.Task.objects.create(name="Auto Fit migration fixture", data=data, dimension="3d")
        segment = models.Segment.objects.create(task=cls.task, start_frame=0, stop_frame=1)
        cls.job = models.Job.objects.create(segment=segment)
        cls.car = models.Label.objects.create(task=cls.task, name="Car", type="cuboid", color="#123456")
        cls.shape_attr = models.AttributeSpec.objects.create(
            label=cls.car, name="paint", mutable=False, input_type="select",
            default_value="red", values="red\nblue",
        )
        cls.mutable_attr = models.AttributeSpec.objects.create(
            label=cls.car, name="speed", mutable=True, input_type="number",
            default_value="0", values="0\n100",
        )
        cls.shape = models.LabeledShape.objects.create(
            job=cls.job, label=cls.car, frame=0, type="cuboid",
            points=[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 4.0, 1.8, 1.5, 0, 0, 0, 0, 0, 0, 0],
        )
        cls.shape_value = models.LabeledShapeAttributeVal.objects.create(
            spec=cls.shape_attr, job=cls.job, value="red", shape=cls.shape,
        )
        cls.shape_speed = models.LabeledShapeAttributeVal.objects.create(
            spec=cls.mutable_attr, job=cls.job, value="27", shape=cls.shape,
        )
        cls.track = models.LabeledTrack.objects.create(job=cls.job, label=cls.car, frame=0)
        cls.track_value = models.LabeledTrackAttributeVal.objects.create(
            spec=cls.shape_attr, job=cls.job, value="blue", track=cls.track,
        )
        cls.keyframe0 = models.TrackedShape.objects.create(
            track=cls.track, frame=0, type="cuboid", points=[0.0] * 16,
        )
        cls.keyframe1 = models.TrackedShape.objects.create(
            track=cls.track, frame=1, type="cuboid", points=[1.0] * 16,
        )
        cls.keyframe_speed = models.TrackedShapeAttributeVal.objects.create(
            spec=cls.mutable_attr, job=cls.job, value="31", shape=cls.keyframe1,
        )

        cls.project = models.Project.objects.create(name="Auto Fit project migration fixture")
        cls.project_car = models.Label.objects.create(
            project=cls.project, name="Car", type="cuboid", color="#654321",
        )
        cls.project_tasks = []
        cls.project_shapes = []
        for index in range(2):
            task_data = models.Data.objects.create(size=1, start_frame=0, stop_frame=0, deleted_frames=[])
            task = models.Task.objects.create(
                project=cls.project, name=f"Project fixture task {index}", data=task_data, dimension="3d",
            )
            segment = models.Segment.objects.create(task=task, start_frame=0, stop_frame=0)
            job = models.Job.objects.create(segment=segment)
            shape = models.LabeledShape.objects.create(
                job=job, label=cls.project_car, frame=0, type="cuboid", points=[float(index)] * 16,
            )
            cls.project_tasks.append(task)
            cls.project_shapes.append(shape)

    def run_command(self, *args, **kwargs):
        return call_command("migrate_autofit_vehicles", "--task-id", str(self.task.id), *args, **kwargs)

    def run_project_command(self, *args, **kwargs):
        return call_command("migrate_autofit_vehicles", "--project-id", str(self.project.id), *args, **kwargs)

    def snapshot(self):
        scope = migration.Command()._scope({"task_id": self.task.id, "project_id": None})
        return migration._capture(scope, scope["source"], scope["target"])

    def test_valid_existing_type_roundtrip_and_repeated_apply(self):
        spec = models.AttributeSpec.objects.create(
            label=self.car, name="type", mutable=False, input_type="select",
            default_value="other", values="\n".join(migration.VEHICLE_TYPES),
        )
        models.LabeledShapeAttributeVal.objects.create(spec=spec, job=self.job, shape=self.shape, value="car")
        models.LabeledTrackAttributeVal.objects.create(spec=spec, job=self.job, track=self.track, value="car")
        before = self.snapshot()
        with tempfile.TemporaryDirectory() as directory:
            backup = str(Path(directory) / "before.json")
            self.run_command("--mode", "backup", "--backup", backup)
            for _ in range(2):
                self.run_command("--mode", "apply", "--backup", backup,
                                 "--confirm-scope", f"task:{self.task.id}")
                target = models.Label.objects.get(task=self.task, name="Vehicles")
                self.assertEqual(target.attributespec_set.filter(name="type").count(), 1)
                self.assertEqual(models.LabeledShapeAttributeVal.objects.filter(
                    shape=self.shape, spec__name="type").count(), 1)
                self.assertEqual(models.LabeledTrackAttributeVal.objects.filter(
                    track=self.track, spec__name="type").count(), 1)
            self.run_command("--mode", "rollback", "--backup", backup)
        self.assertEqual(self.snapshot(), before)

    def test_invalid_existing_type_is_rejected_without_changes(self):
        spec = models.AttributeSpec.objects.create(
            label=self.car, name="type", mutable=False, input_type="text", default_value="", values="",
        )
        for changes in ({}, {"input_type": "select", "values": "car"},
                        {"values": "\n".join(migration.VEHICLE_TYPES), "mutable": True}):
            for name, value in changes.items():
                setattr(spec, name, value)
            spec.save()
            before = self.snapshot()
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(CommandError, "Car/type conflicts"):
                    self.run_command("--mode", "backup", "--backup", str(Path(directory) / "before.json"))
                self.assertEqual(self.snapshot(), before)

    def check_tampered_backup(self, mode, applied=False):
        with tempfile.TemporaryDirectory() as directory:
            backup = Path(directory) / "before.json"
            self.run_command("--mode", "backup", "--backup", str(backup))
            if applied:
                self.run_command("--mode", "apply", "--backup", str(backup),
                                 "--confirm-scope", f"task:{self.task.id}")
            data = json.loads(backup.read_text())
            data["before"]["shapes"][0]["points"][0] += 100
            backup.write_text(json.dumps(data))
            contents, before = backup.read_bytes(), self.snapshot()
            with patch.object(migration.Command, "_apply", wraps=migration.Command()._apply) as apply, \
                    patch.object(migration, "_restore_rows", wraps=migration._restore_rows) as restore:
                with self.assertRaisesRegex(CommandError, "checksum"):
                    self.run_command("--mode", mode, "--backup", str(backup),
                                     "--confirm-scope", f"task:{self.task.id}")
                apply.assert_not_called()
                restore.assert_not_called()
            self.assertEqual(self.snapshot(), before)
            self.assertEqual(backup.read_bytes(), contents)

    def test_tampered_backup_rejected_before_apply(self):
        self.check_tampered_backup("apply")

    def test_tampered_backup_rejected_before_repeated_apply(self):
        self.check_tampered_backup("apply", applied=True)

    def test_tampered_backup_rejected_before_rollback(self):
        self.check_tampered_backup("rollback", applied=True)

    def test_backup_apply_and_rollback_preserve_shape_track_keyframes_and_attributes(self):
        with tempfile.TemporaryDirectory() as directory:
            backup = Path(directory) / "vehicle-migration.json"
            self.run_command("--mode", "dry-run")
            with self.assertRaises(CommandError):
                self.run_command("--mode", "apply", "--backup", str(backup),
                                 "--confirm-scope", f"task:{self.task.id}")
            self.run_command("--mode", "backup", "--backup", str(backup))
            original = json.loads(backup.read_text())
            self.assertEqual(original["status"], "backed-up")

            self.run_command("--mode", "apply", "--backup", str(backup),
                             "--confirm-scope", f"task:{self.task.id}")
            vehicle = models.Label.objects.get(task=self.task, name="Vehicles")
            type_spec = vehicle.attributespec_set.get(name="type")
            self.shape.refresh_from_db()
            self.track.refresh_from_db()
            self.keyframe0.refresh_from_db()
            self.keyframe1.refresh_from_db()
            self.assertEqual((self.shape.id, self.shape.label_id, self.shape.points),
                             (original["before"]["shapes"][0]["id"], vehicle.id,
                              original["before"]["shapes"][0]["points"]))
            self.assertEqual((self.track.id, self.track.label_id),
                             (original["before"]["tracks"][0]["id"], vehicle.id))
            self.assertEqual([self.keyframe0.id, self.keyframe1.id],
                             [row["id"] for row in original["before"]["tracked_shapes"]])
            self.assertEqual(models.LabeledShapeAttributeVal.objects.get(shape=self.shape, spec=type_spec).value,
                             "car")
            self.assertEqual(models.LabeledTrackAttributeVal.objects.get(track=self.track, spec=type_spec).value,
                             "car")
            self.assertEqual(models.TrackedShapeAttributeVal.objects.get(
                shape=self.keyframe1, spec__name="speed", spec__label=vehicle,
            ).value, "31")
            self.assertEqual(models.LabeledShapeAttributeVal.objects.get(
                shape=self.shape, spec__name="paint", spec__label=vehicle,
            ).value, "red")

            applied = self.snapshot()
            self.run_command("--mode", "apply", "--backup", str(backup),
                             "--confirm-scope", f"task:{self.task.id}")
            self.assertEqual(self.snapshot(), applied)
            self.run_command("--mode", "rollback", "--backup", str(backup))
            self.shape.refresh_from_db()
            self.track.refresh_from_db()
            self.keyframe1.refresh_from_db()
            self.assertEqual(self.shape.label_id, self.car.id)
            self.assertEqual(self.shape.points, original["before"]["shapes"][0]["points"])
            self.assertEqual(self.track.label_id, self.car.id)
            self.assertEqual(self.keyframe1.points, original["before"]["tracked_shapes"][1]["points"])
            self.assertEqual(models.LabeledShapeAttributeVal.objects.get(pk=self.shape_value.id).value, "red")
            self.assertEqual(models.LabeledShapeAttributeVal.objects.get(pk=self.shape_speed.id).value, "27")
            self.assertEqual(models.LabeledTrackAttributeVal.objects.get(pk=self.track_value.id).value, "blue")
            self.assertEqual(models.TrackedShapeAttributeVal.objects.get(pk=self.keyframe_speed.id).value, "31")
            self.assertFalse(models.Label.objects.filter(task=self.task, name="Vehicles").exists())

    def test_rollback_stops_if_annotations_changed_after_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            backup = Path(directory) / "vehicle-migration.json"
            self.run_command("--mode", "backup", "--backup", str(backup))
            self.run_command("--mode", "apply", "--backup", str(backup),
                             "--confirm-scope", f"task:{self.task.id}")
            self.shape.points = [2.0] * 16
            self.shape.save(update_fields=["points"])
            with self.assertRaisesRegex(CommandError, "changed after migration"):
                self.run_command("--mode", "rollback", "--backup", str(backup))
            self.shape.refresh_from_db()
            self.assertEqual(self.shape.points, [2.0] * 16)

    def test_project_scope_migrates_each_project_task_and_rolls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            backup = Path(directory) / "project-before.json"
            self.run_project_command("--mode", "backup", "--backup", str(backup))
            self.run_project_command("--mode", "apply", "--backup", str(backup),
                                     "--confirm-scope", f"project:{self.project.id}")
            vehicle = models.Label.objects.get(project=self.project, name="Vehicles")
            type_spec = vehicle.attributespec_set.get(name="type")
            for shape in self.project_shapes:
                shape.refresh_from_db()
                self.assertEqual(shape.label_id, vehicle.id)
                self.assertEqual(models.LabeledShapeAttributeVal.objects.get(
                    shape=shape, spec=type_spec,
                ).value, "car")
            self.run_project_command("--mode", "rollback", "--backup", str(backup))
            for shape in self.project_shapes:
                shape.refresh_from_db()
                self.assertEqual(shape.label_id, self.project_car.id)
            self.assertFalse(models.Label.objects.filter(project=self.project, name="Vehicles").exists())
