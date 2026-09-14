"""Terminal checklist for the numbered exercise TODOs (standard library only)."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / '.todo-checklist.json'


def scan(items):
    # Migrate the original numbering after inserting attention as TODOs 4–5.
    # The old constructor title identifies the old layout without changing
    # the state file format or migrating the same checklist twice.
    if items.get('4', {}).get('title') == 'Construct one pre-normalization transformer block.':
        items = {
            str(int(number) + 2 if int(number) >= 4 else int(number)): item
            for number, item in items.items()
        }
    # Retain completed/removed comments so the checklist keeps its history.
    for path in sorted((ROOT / 'src').rglob('*.py')):
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            match = re.match(r'\s*# TODO (\d+):\s*(.*)', line)
            if not match:
                continue
            number, title = match.groups()
            if not title and index + 1 < len(lines):
                title = lines[index + 1].strip().removeprefix('#').strip()
            previous = items.get(number, {})
            items[number] = dict(previous, title=title, path=str(path.relative_to(ROOT)), line=index + 1)
    return items


def save(items):
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(items, indent=2) + '\n')
    temporary.replace(STATE)


def show(items):
    styled = sys.stdout.isatty() and 'NO_COLOR' not in os.environ
    done = sum(bool(item.get('done')) for item in items.values())
    print(f'\nExercise TODOs — {done}/{len(items)} complete\n')
    for number in sorted(items, key=int):
        item = items[number]
        label = f"[{ 'x' if item.get('done') else ' ' }] {int(number):2}. {item['title']}"
        if styled and item.get('done'):
            label = f'\033[9;2m{label}\033[0m'
        # Let the IDE terminal recognize the path and open it in that IDE.
        # An OSC hyperlink with vscode:// would force Cursor users into VS Code.
        location = f"{ROOT / item['path']}:{item['line']}"
        print(f'{label}\n        {location}')


def open_item(item):
    path = str(ROOT / item['path'])
    line = str(item['line'])
    editor = os.environ.get('TODO_EDITOR') or os.environ.get('VISUAL') or os.environ.get('EDITOR')
    if not editor:
        editor = next((name for name in ('code', 'cursor', 'nvim', 'vim', 'vi') if shutil.which(name)), None)
    if not editor:
        print('Set TODO_EDITOR to your editor command, then try again.')
        return
    command = shlex.split(editor)
    name = Path(command[0]).name
    if name in ('code', 'code-insiders', 'cursor'):
        command += ['--goto', f'{path}:{line}']
    elif name in ('subl', 'mate'):
        command += [f'{path}:{line}']
    else:
        command += [f'+{line}', path]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f'Could not open editor: {error}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='Print checklist without prompting or saving state')
    args = parser.parse_args()
    try:
        items = json.loads(STATE.read_text()) if STATE.exists() else {}
    except (OSError, ValueError) as error:
        parser.exit(1, f'Cannot read {STATE}: {error}\n')
    items = scan(items)
    if args.list:
        show(items)
        return
    save(items)
    while True:
        show(items)
        print('\nEnter a number to open; d NUMBER to toggle done; r to refresh; q to quit.')
        print('Cmd-click a file path to open it in your current IDE (Ctrl-click on Linux/Windows).')
        try:
            command = input('todo> ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if command == 'q':
            return
        if command in ('', 'r'):
            items = scan(items)
            save(items)
            continue
        toggle = command.startswith('d ')
        number = command[2:].strip() if toggle else command
        number = str(int(number)) if number.isdecimal() else number
        if number not in items:
            print('Use a listed TODO number, d NUMBER, r, or q.')
            continue
        if toggle:
            items[number]['done'] = not items[number].get('done', False)
            save(items)
        else:
            open_item(items[number])
            items = scan(items)
            save(items)


if __name__ == '__main__':
    main()
