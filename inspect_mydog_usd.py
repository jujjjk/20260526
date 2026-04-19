from pxr import Usd, UsdPhysics

usd_path = "/home/nszb/python_text/robot_assets/mydog_description/usd/mydog.usd"
stage = Usd.Stage.Open(usd_path)

if stage is None:
    print("打开 USD 失败")
    raise SystemExit(1)

default_prim = stage.GetDefaultPrim()
print("USD path:", usd_path)
print("DefaultPrim:", default_prim.GetPath() if default_prim else "None")

print("\n=== Joints ===")
for prim in stage.Traverse():
    t = prim.GetTypeName()
    if "Joint" in t:
        print(f"{prim.GetPath()}   [{t}]")

print("\n=== Rigid Bodies ===")
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print(f"{prim.GetPath()}   [{prim.GetTypeName()}]")
