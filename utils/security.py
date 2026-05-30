import uuid
import bcrypt
from pathlib import Path
import config

DEVICE_ID_FILE = Path(config.SQLITE_DB_PATH).parent / ".device_id"


# ── Get Device ID

def get_device_id() -> str:
    
    if DEVICE_ID_FILE.exists():
        did = DEVICE_ID_FILE.read_text().strip()
        if did:
            return did

    # First run on this machine — generate and persist
    did = str(uuid.uuid4())
    DEVICE_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEVICE_ID_FILE.write_text(did)
    return did


# ── PIN utilities 

def hash_pin(pin: str) -> str:

    pin_bytes = pin.strip().encode("utf-8")
    return bcrypt.hashpw(pin_bytes, bcrypt.gensalt(rounds=10)).decode("utf-8")


def verify_pin(pin: str, pin_hash: str) -> bool:

    try:
        return bcrypt.checkpw(
            pin.strip().encode("utf-8"),
            pin_hash.encode("utf-8"),
        )
    except Exception:
        return False


def validate_pin_format(pin: str) -> tuple[bool, str]:

    pin = pin.strip()
    if len(pin) != 4:
        return False, "PIN must be exactly 4 digits"
    if not pin.isdigit():
        return False, "PIN must contain only numbers"
    return True, ""