"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Copied from UNITER
(https://github.com/ChenRocks/UNITER)
"""
from .sched import noam_schedule, warmup_linear, vqa_schedule, get_lr_sched
from .adamw import AdamW