from typing import List, Tuple, Any
from collections import defaultdict
from robot_utils import console


def create_dependency_graph(package_list):
    dependency_graph = defaultdict(list)
    for package, dependencies in package_list:
        dependency_graph[package].extend(dependencies)
    return dependency_graph


def get_install_order(
        package_list: List[Tuple[str, List[str]]]
) -> List[str]:
    dependency_graph = create_dependency_graph(package_list)
    visited = set()
    result = []

    def dfs(package):
        visited.add(package)
        dependencies = list(dependency_graph[package])  # Create a copy of dependencies
        for dependency in dependencies:
            if dependency not in visited:
                dfs(dependency)
        result.append(package)

    for pkg in list(dependency_graph.keys()):  # Iterate over a copy of keys
        if pkg not in visited:
            dfs(pkg)

    return result


if __name__ == "__main__":
    pkg_list = [
        ("PackageA", ["PackageB", "PackageC"]),
        ("PackageB", []),
        ("PackageC", ["PackageD"]),
        ("PackageD", ['PackageB']),
        ("PackageE", ["PackageB", "PackageD", "PackageF"])
    ]
    ic(topological_sort(pkg_list))
