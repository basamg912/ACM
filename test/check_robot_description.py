import argparse


armature = [0.01936, 0.01936, 0.01936, 0.01936, 0.01936, 0.00565, 0.01936, 0.01936, 0.01936, 0.01936, 0.01936, 0.00565, 0.01936, 0.00565, 0.00565, 0.00565, 0.00565, 0.00565, 0.00565, 0.00565, 0.00565, 0.00565, 0.00565]
p_gain = [76.447, 76.447, 76.447, 76.447, 76.447, 22.3, 76.447, 76.447, 76.447, 76.447, 76.447, 22.3, 76.447, 22.3, 22.3, 22.3, 22.3, 22.3, 22.3, 22.3, 22.3, 22.3, 22.3]
d_gain = [4.862, 4.862, 4.862, 4.862, 4.862, 1.418, 4.862, 4.862, 4.862, 4.862, 4.862, 1.418, 4.862, 1.418, 1.418, 1.418, 1.418, 1.418, 1.418, 1.418, 1.418, 1.418, 1.418]

parser = argparse.ArgumentParser()
parser.add_argument("--path",help="robot's URDF description path", required=True)
args = parser.parse_args()

import xml.etree.ElementTree as ET

def parse_robot(path: str):
    tree = ET.parse(args.path)
    return tree.getroot()

def find_link(root, verbose=True):
    links = root.findall("link")
    if verbose:
        print("========================")
        print("Link : ",len(links))
        for link in links:
            print(link.get("name"))
        print("========================")
    return links

def find_joint(root, verbose=True, **kwargs):
    joints = root.findall("joint")
    dynamics = kwargs.get("dynamics", False)
    if verbose:
        print("========================")
        print("Joint : ",len(joints))
        for joint in joints:
            print(joint.get("name"))
            if dynamics:
                d_tag = joint.find("dynamics")
                print(d_tag)
        print("========================")
    return joints

def find_limit(targets, verbose=True, **kwargs):
    print_option = kwargs.get("print_option", "joint")
    if print_option == "joint":
        for target in targets:
            t_name = target.get("name")
            t_type = target.get("type")
            limit_tag = target.find("limit")
            if limit_tag is not None:
                lower = limit_tag.get("lower")
                upper = limit_tag.get("upper")
                effort = limit_tag.get("effort")
                velocity = limit_tag.get("velocity")
                print(f"Name: {t_name} / Type: {t_type}")
                print(f"{lower} {upper} {effort} {velocity}")
    elif print_option == "limit":
        import pandas as pd
        names = [
            str(target.get("name"))
            for target in targets
        ]
        lower = [
            float(target.find('limit').get('lower'))
            if target.find('limit') is not None and target.find('limit').get('lower') is not None
            else None
            for target in targets
        ]
        upper = [
            float(target.find('limit').get('upper'))
            if target.find('limit') is not None and target.find('limit').get('upper') is not None
            else None
            for target in targets
        ]
        effort = [
            float(target.find('limit').get('effort'))
            if target.find('limit') is not None and target.find('limit').get('effort') is not None
            else None
            for target in targets
        ]
        velocity = [
            float(target.find('limit').get('velocity'))
            if target.find('limit') is not None and target.find('limit').get('velocity') is not None
            else None
            for target in targets
        ]
        df = pd.DataFrame(
            {"lower" : lower, "upper" : upper, "effort": effort, "velocity": velocity},
            index=pd.Index(names)
        )
    if verbose:
        print("========================")
        print(df)
        print("========================")
    elif verbose == False:
        return lower, upper, effort, velocity

root = parse_robot(args.path)
find_link(root)
# find_joint(root, dynamics=True)
# lower, upper, effort, velocity = find_limit(find_joint(root,False), verbose=False, print_option="limit")
# print(f"lower: {lower}")
# print(f"upper: {upper}")
# print(f"effort: {effort}")
# print(f"velocity: {velocity}")
# print(f"p_gain ({len(p_gain)}): {p_gain}")
# print(f"d_gain ({len(d_gain)}): {d_gain}")
# print(f"armature ({len(armature)}): {armature}")
