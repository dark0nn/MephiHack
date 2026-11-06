"""Чекпоинты: сохранение полного состояния и загрузка весов в изменённую сеть.

Что лежит в .pt:
    model  — веса U-Net (state_dict)
    opt    — состояние оптимизатора (моменты Adam), sched — расписание lr
    step   — номер шага, best — лучший val loss, val — метрики, args — параметры запуска

Два способа использовать чекпоинт (см. scripts/train.py):
    --continue  продолжить тот же запуск с того же шага (после обрыва, выключенного ноутбука и т.п.)
    --resume    взять веса как старт для новой версии модели; совпадающие слои загрузятся,
                новые/изменённые слои начнут со случайной инициализации
"""
import torch
import torch.nn as nn


def save_checkpoint(path, model, opt, sched, step, best, val, args):
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "step": step,
        "best": best,
        "val": val,
        "args": dict(args),
    }, path)


def load_matching(model: nn.Module, state: dict, verbose: bool = True) -> dict:
    """Грузит из state только те тензоры, у которых совпадает имя И форма.

    Возвращает {"loaded": [...], "new": [...], "changed": [...], "dropped": [...]}:
      loaded  — перенесены из чекпоинта
      new     — есть в модели, нет в чекпоинте (новые слои) -> случайная инициализация
      changed — имя то же, форма другая (поменяли число каналов) -> случайная инициализация
      dropped — были в чекпоинте, в модели их больше нет
    """
    own = model.state_dict()
    keep, changed = {}, []
    for k, v in state.items():
        if k in own:
            if own[k].shape == v.shape:
                keep[k] = v
            else:
                changed.append(k)
    dropped = [k for k in state if k not in own]
    new = [k for k in own if k not in state]
    model.load_state_dict(keep, strict=False)
    rep = {"loaded": list(keep), "new": new, "changed": changed, "dropped": dropped}
    if verbose:
        print(f"Веса из чекпоинта: перенесено {len(keep)} из {len(own)} тензоров модели")
        for name, title in (("new", "новые слои (с нуля)"), ("changed", "изменилась форма (с нуля)"),
                            ("dropped", "есть в чекпоинте, нет в модели (пропущены)")):
            if rep[name]:
                shown = ", ".join(rep[name][:6]) + (" ..." if len(rep[name]) > 6 else "")
                print(f"  {title}: {len(rep[name])} — {shown}")
    return rep
