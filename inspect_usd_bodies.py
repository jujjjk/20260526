import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect rigid bodies and joints in a USD file.")
parser.add_argument("usd_path", type=str, help="Path to the USD file.")
parser.add_argument(
    "--show-joints",
    action="store_true",
    help="Also print joint prims found in the USD.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics


def main() -> int:
    usd_path = Path(args_cli.usd_path).expanduser().resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        print(f"Failed to open USD: {usd_path}")
        return 1

    print(f"USD path: {usd_path}")
    default_prim = stage.GetDefaultPrim()
    print(f"DefaultPrim: {default_prim.GetPath() if default_prim else 'None'}")

    rigid_bodies: list[str] = []
    joints: list[str] = []

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(prim.GetName())
        if args_cli.show_joints and "Joint" in prim.GetTypeName():
            joints.append(f"{prim.GetPath()} [{prim.GetTypeName()}]")

    print("\nRigid bodies:")
    for name in rigid_bodies:
        print(f"  {name}")

    if args_cli.show_joints:
        print("\nJoints:")
        for item in joints:
            print(f"  {item}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
