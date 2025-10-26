from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

ProbeSource = dict[str, Any]


@dataclass(slots=True)
class ProbeTask:
    """Single probe configuration loaded from the JSONL manifest."""

    task_id: str
    domain: str
    description: str
    source: ProbeSource
    max_tokens: int | None = None
    repeat: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProbeTask:
        return cls(
            task_id=payload['task_id'],
            domain=payload['domain'],
            description=payload.get('description', ''),
            source=payload['source'],
            max_tokens=payload.get('max_tokens'),
            repeat=payload.get('repeat'),
        )

    def iter_texts(self) -> Iterator[str]:
        kind = self.source.get('type')
        if kind == 'prompt':
            text = self.source['text']
            total = self.source.get('repeat', self.repeat or 1)
            for _ in range(total):
                yield text
            return

        if kind == 'prompt_list':
            prompts: tuple[str, ...] = tuple(self.source['prompts'])
            total = self.source.get('repeat', self.repeat or 1)
            for _ in range(total):
                yield from prompts
            return

        if kind == 'dataset':
            name = self.source['name']
            config = self.source.get('config')
            split = self.source.get('split', 'train')
            text_key = self.source.get('text_key', 'text')
            limit = self.source.get('num_docs', self.repeat)

            dataset = load_dataset(name, config, split=split)
            for idx, sample in enumerate(dataset):
                if limit is not None and idx >= limit:
                    break
                if text_key in sample:
                    yield sample[text_key]
                else:
                    yield str(sample)
            return

        raise ValueError(f'Unsupported probe source type: {kind!r}')

    def metadata(self) -> dict[str, Any]:
        source_meta = {k: v for k, v in self.source.items() if k != 'prompts' and k != 'text'}
        meta: dict[str, Any] = {
            'task_id': self.task_id,
            'domain': self.domain,
            'description': self.description,
            'source': {'type': self.source.get('type'), **source_meta},
        }
        if self.max_tokens is not None:
            meta['max_tokens'] = self.max_tokens
        if self.repeat is not None:
            meta['repeat'] = self.repeat
        return meta


def load_probe_tasks(path: Path) -> dict[str, ProbeTask]:
    tasks: dict[str, ProbeTask] = {}
    with path.open() as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith('#'):
                continue
            data = json.loads(stripped)
            task = ProbeTask.from_dict(data)
            if task.task_id in tasks:
                raise ValueError(
                    f'duplicate task_id {task.task_id!r} in manifest {path} (line {line_number})'
                )
            tasks[task.task_id] = task
    return tasks


def select_probe_tasks(tasks: dict[str, ProbeTask], ids: Iterable[str] | None) -> list[ProbeTask]:
    if ids is None:
        return list(tasks.values())
    selected: list[ProbeTask] = []
    for task_id in ids:
        if task_id not in tasks:
            raise KeyError(f'Task {task_id!r} not found in manifest')
        selected.append(tasks[task_id])
    return selected
