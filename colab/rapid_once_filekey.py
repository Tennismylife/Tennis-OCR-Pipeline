#!/usr/bin/env python3
import argparse, base64, csv, subprocess, sys
from collections import defaultdict
from pathlib import Path

from worker import connect_sftp, read_manifest, pg


def write_subset(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description='One-shot RapidOCR batch for interactive/free Colab use')
    ap.add_argument('--vps-key-file', required=True)
    ap.add_argument('--vps-host', required=True)
    ap.add_argument('--vps-user', required=True)
    ap.add_argument('--vps-port', type=int, default=2222)
    ap.add_argument('--claim', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--downloaders', type=int, default=4)
    ap.add_argument('--workdir', default='/content/tml_rapid_once')
    a = ap.parse_args()

    key = Path(a.vps_key_file)
    if not key.exists() or key.stat().st_size < 100:
        raise SystemExit(f'Invalid key file: {key}')
    token = base64.b64encode(key.read_bytes()).decode()

    wd = Path(a.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    local_claim = wd / 'claim.tsv'

    tr, sftp = connect_sftp(a.vps_host, a.vps_user, token, a.vps_port)
    try:
        sftp.get(a.claim, str(local_claim))
        rows = [r for r in read_manifest(str(local_claim)) if (r.get('mode') or r.get('split') or '').upper() == 'RAPID']
        groups = defaultdict(list)
        for row in rows:
            cache = (row.get('remote_cache') or '').strip()
            if not cache:
                raise RuntimeError('Claim row missing remote_cache')
            groups[cache].append(row)

        incomplete = {}
        missing = 0
        for cache, grows in groups.items():
            try:
                names = set(sftp.listdir(cache.rstrip('/')))
            except IOError:
                names = set()
            todo = []
            for row in grows:
                ark = (row.get('ark') or '').strip()
                page = pg(row.get('page'))
                stem = f'{ark}_f{page}'
                if f'{stem}.json' not in names or f'{stem}.txt' not in names:
                    todo.append(row)
            incomplete[cache] = todo
            missing += len(todo)
    finally:
        sftp.close()
        tr.close()

    print(f'FREE_RAPID_ONCE claim_rows={len(rows)} groups={len(groups)} missing={missing}', flush=True)
    if not rows or missing == 0:
        print('FREE_RAPID_ONCE_COMPLETE nothing_to_do=1', flush=True)
        return 0

    rc_total = 0
    for idx, (cache, grows) in enumerate(incomplete.items(), 1):
        if not grows:
            continue
        subset = wd / f'group_{idx}.tsv'
        write_subset(subset, grows)
        profile = grows[0].get('split_profile') or 'HQ'
        print(f'FREE_RAPID_GROUP {idx}/{len(incomplete)} rows={len(grows)} cache={cache}', flush=True)
        cmd = [
            sys.executable, '-u', str(Path(__file__).with_name('rapid_pool.py')),
            '--manifest', str(subset),
            '--vps-host', a.vps_host,
            '--vps-user', a.vps_user,
            '--vps-key-b64', token,
            '--vps-port', str(a.vps_port),
            '--remote-cache', cache,
            '--profile', profile,
            '--workers', str(a.workers),
            '--downloaders', str(a.downloaders),
            '--no-autotune',
        ]
        rc = subprocess.run(cmd).returncode
        print(f'FREE_RAPID_GROUP_EXIT group={idx} rc={rc}', flush=True)
        if rc != 0:
            rc_total = rc
            break

    if rc_total == 0:
        print('FREE_RAPID_ONCE_COMPLETE success=1', flush=True)
    return rc_total


if __name__ == '__main__':
    raise SystemExit(main())
