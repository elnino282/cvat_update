# Copyright (C) CVAT.ai Corporation
# SPDX-License-Identifier: MIT

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("engine", "0108_profile_registration_survey_fields")]

    operations = [
        migrations.AddField(
            model_name="labeledshape",
            name="auto_fit",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="labeledtrack",
            name="auto_fit",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
