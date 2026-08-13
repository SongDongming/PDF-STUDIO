from fastapi import HTTPException, status

from app.store import store


def require_item(collection: str, item_id: str, label: str) -> dict:
    item = store.get(collection, item_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": f"{label}不存在"},
        )
    return item

