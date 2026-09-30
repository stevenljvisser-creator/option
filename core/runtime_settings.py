from pathlib import Path
from functools import lru_cache
from cryptography.fernet import Fernet
from sqlalchemy import select
from core.db import SessionLocal
from core.models import AppSetting
from core.config import SETTINGS_KEY_FILE, DEFAULT_IMPORT_CHUNK_ROWS

SECRET_KEYS={
    "HETZNER_S3_ACCESS_KEY",
    "HETZNER_S3_SECRET_KEY",
    "MASSIVE_API_KEY",
    "MASSIVE_FLAT_ACCESS_KEY",
    "MASSIVE_FLAT_SECRET_KEY",
    "FMP_API_KEY",
}

REQUIRED_KEYS=[
    "HETZNER_S3_ENDPOINT",
    "HETZNER_S3_REGION",
    "HETZNER_S3_BUCKET",
    "HETZNER_S3_ACCESS_KEY",
    "HETZNER_S3_SECRET_KEY",
    "MASSIVE_API_KEY",
    "MASSIVE_FLAT_ACCESS_KEY",
    "MASSIVE_FLAT_SECRET_KEY",
]

def _fernet():
    path=Path(SETTINGS_KEY_FILE)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        path.write_bytes(Fernet.generate_key())
        try:
            path.chmod(0o600)
        except Exception:
            pass
    return Fernet(path.read_bytes().strip())

def encrypt_value(value:str)->str:
    return _fernet().encrypt((value or "").encode("utf-8")).decode("ascii")

def decrypt_value(value:str)->str:
    return _fernet().decrypt(value.encode("ascii")).decode("utf-8")

def save_settings(values:dict):
    db=SessionLocal()
    try:
        for key,val in values.items():
            if val is None:
                continue
            val=str(val)
            row=db.get(AppSetting,key) or AppSetting(key=key)
            row.encrypted=key in SECRET_KEYS
            row.value=encrypt_value(val) if row.encrypted else val
            db.add(row)
        db.commit()
    finally:
        db.close()

def get_setting(key:str,default=None):
    db=SessionLocal()
    try:
        row=db.get(AppSetting,key)
        if not row:
            return default
        return decrypt_value(row.value) if row.encrypted else row.value
    finally:
        db.close()

def get_storage_settings():
    """Read only the storage configuration in one database snapshot."""
    keys=("HETZNER_S3_ENDPOINT","HETZNER_S3_REGION","HETZNER_S3_BUCKET",
          "HETZNER_S3_ACCESS_KEY","HETZNER_S3_SECRET_KEY")
    db=SessionLocal()
    try:
        rows=db.scalars(select(AppSetting).where(AppSetting.key.in_(keys))).all()
        return {
            row.key:decrypt_value(row.value) if row.encrypted else row.value
            for row in rows
        }
    finally:
        db.close()

def get_settings(mask_secrets=False):
    db=SessionLocal()
    try:
        rows=db.scalars(select(AppSetting)).all()
        out={}
        for row in rows:
            if row.encrypted:
                raw=decrypt_value(row.value)
                out[row.key]=("••••••••" if mask_secrets and raw else raw)
            else:
                out[row.key]=row.value
        return out
    finally:
        db.close()

def setup_complete():
    s=get_settings(mask_secrets=False)
    return all(bool(s.get(k)) for k in REQUIRED_KEYS)

def import_chunk_rows():
    try:
        return max(50000,min(600000,int(get_setting("IMPORT_CHUNK_ROWS",DEFAULT_IMPORT_CHUNK_ROWS))))
    except Exception:
        return DEFAULT_IMPORT_CHUNK_ROWS
