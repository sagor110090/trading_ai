import os
os.environ.setdefault("USE_SQLITE", "true")

from trading_ai.data.database import get_session
from trading_ai.data.models import MLModelMeta

with get_session() as session:
    models = session.query(MLModelMeta).filter(
        MLModelMeta.symbol == 'BTCUSD'
    ).order_by(MLModelMeta.version.desc()).all()

    if models:
        print("BTCUSD ML Models:")
        for m in models[:5]:
            print(f'  {m.model_type}_v{m.version} - active={m.is_active} - auc={m.roc_auc}')
    else:
        print('No models found for BTCUSD')

    # Also check for all symbols
    print("\nAll ML Models:")
    all_models = session.query(MLModelMeta).filter(MLModelMeta.is_active == True).all()
    for m in all_models:
        print(f'  {m.symbol} - {m.model_type}_v{m.version} - active={m.is_active}')