# Copyright (C) CVAT.ai Corporation
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from cvat.apps.dataset_manager.task import JobAnnotation
from cvat.apps.engine.serializers import (
    LabeledDataSerializer, LabeledShapeSerializerFromDB, LabeledTrackSerializerFromDB,
)
from cvat.apps.engine.tests.test_auto_fit_metadata import METADATA


class AutoFitMetadataSerializerTests(SimpleTestCase):
    def shape(self):
        return {
            "id": 1, "label_id": 7, "frame": 0, "group": 0, "source": "auto",
            "type": "cuboid", "points": [1.0] * 16, "rotation": 0, "z_order": 0,
            "occluded": False, "outside": False, "attributes": [], "elements": [],
            "auto_fit": dict(METADATA), "score": 1.0,
        }

    def track(self):
        shape = self.shape()
        return {
            "id": 2, "label_id": 7, "frame": 0, "group": 0, "source": "auto",
            "attributes": [], "elements": [], "auto_fit": dict(METADATA),
            "shapes": [{key: shape[key] for key in (
                "id", "type", "frame", "points", "rotation", "z_order",
                "occluded", "outside", "attributes",
            )}],
        }

    def test_public_shape_track_contract_preserves_creation_metadata(self):
        data = {"shapes": [self.shape()], "tracks": [self.track()]}
        serializer = LabeledDataSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        for key in ("shapes", "tracks"):
            self.assertEqual(serializer.validated_data[key][0]["auto_fit"], METADATA)
        self.assertNotIn("auto_fit", serializer.validated_data["tracks"][0]["shapes"][0])

    def test_invalid_metadata_rejected_and_old_clients_can_omit_it(self):
        for key, factory in (("shapes", self.shape), ("tracks", self.track)):
            with self.subTest(key=key):
                item = factory()
                item["auto_fit"]["metrics"] = {"point_count": 10}
                serializer = LabeledDataSerializer(data={key: [item]})
                self.assertFalse(serializer.is_valid())
                self.assertIn("auto_fit", serializer.errors[key][0])
                del item["auto_fit"]
                serializer = LabeledDataSerializer(data={key: [item]})
                self.assertTrue(serializer.is_valid(), serializer.errors)
                self.assertNotIn("auto_fit", serializer.validated_data[key][0])

    def test_fast_db_serializers_return_metadata_without_copying_it_to_keyframes(self):
        shape = self.shape()
        shape["parent"] = None
        track = self.track()
        track["parent"] = None
        self.assertEqual(LabeledShapeSerializerFromDB(shape).data["auto_fit"], METADATA)
        result = LabeledTrackSerializerFromDB(track).data
        self.assertEqual(result["auto_fit"], METADATA)
        self.assertNotIn("auto_fit", result["shapes"][0])

    @patch("cvat.apps.dataset_manager.task.handle_annotations_change")
    def test_geometry_update_from_older_client_keeps_db_provenance(self, _change):
        class Data(dict):
            @property
            def data(self):
                return self

        data = Data(shapes=[self.shape()], tracks=[self.track()], tags=[], intervals=[])
        for key in ("shapes", "tracks"):
            del data[key][0]["auto_fit"]
        shape_manager = MagicMock()
        track_manager = MagicMock()
        shape_manager.filter.return_value.values_list.return_value = [(1, dict(METADATA))]
        track_manager.filter.return_value.values_list.return_value = [(2, dict(METADATA))]
        annotation = SimpleNamespace(
            db_job=SimpleNamespace(labeledshape_set=shape_manager, labeledtrack_set=track_manager),
            _validate_input_annotations=lambda value: value,
            _delete=MagicMock(return_value=data), _create=MagicMock(), data=data,
            _data_is_empty=lambda value: True,
        )
        annotation._preserve_auto_fit_metadata = lambda value: JobAnnotation._preserve_auto_fit_metadata(
            annotation, value
        )
        for method in (JobAnnotation.update, JobAnnotation.put):
            for key in ("shapes", "tracks"):
                data[key][0].pop("auto_fit", None)
            method(annotation, data)
            saved = annotation._create.call_args.args[0]
            self.assertEqual(saved["shapes"][0]["auto_fit"], METADATA)
            self.assertEqual(saved["tracks"][0]["auto_fit"], METADATA)
