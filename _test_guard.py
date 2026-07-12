from modules._s2_legacy_functions import _category_guard_allows, _soft_asset_guard_allows
tests = [
    ('magazine','0_kite_shield_2k_packed','rifle',[0.3,0.2,0.1],False),
    ('magazine','0_SM_Magazine_1','magazine',[0.2,0.15,0.01],True),
    ('magazine','a_SM_handgun_02_mag','handgun',[0.3,0.2,0.05],False),
    ('bowl','17_SM_Plates','plate',[0.2,0.2,0.05],False),
    ('bowl','0_SM_Deco020','bowl',[0.15,0.15,0.05],True),
    ('bowl','22_SM_Street_Vendor_Cart_NN_01n','table',[1.0,0.8,0.6],False),
    ('cup','44_sk118_WineGlass01','wine_glass',[0.1,0.1,0.1],False),
    ('cup','a_SM_coffee_cup_01','cup',[0.08,0.08,0.06],True),
    ('storage_shelf','19_SM_TV_Table','tv_cabinet',[0.5,0.4,0.3],False),
    ('storage_shelf','0_steel_frame_shelves_03_2k_packed','storage_rack',[0.6,0.4,0.4],True),
    ('bedside_table','0_SM_Kitchen_top_60cm','kitchen_cabinet',[0.5,0.4,0.3],False),
    ('bedside_table','0_vintage_wooden_drawer_01_2k_packed','nightstand',[0.5,0.4,0.3],True),
    ('chair','0_SM_Chair_Sec001','backrest_chair',[0.5,0.5,0.4],True),
    ('pillow','a_SM_Armchair.001','pillow',[0.8,0.8,0.4],False),
]
all_ok = True
for item,asset,aclass,size,exp in tests:
    ok1,r1 = _soft_asset_guard_allows(item,asset,aclass,size)
    ok2,r2 = _category_guard_allows(item,asset,aclass,size)
    res = ok1 and ok2
    r = r2 if ok1 else r1
    status = 'OK' if res == exp else 'FAIL'
    if res != exp: all_ok = False
    print(f'{status} {item}: expect={exp} got={res} reason={r}')
print(f'{"ALL PASS" if all_ok else "SOME FAILED"}')
