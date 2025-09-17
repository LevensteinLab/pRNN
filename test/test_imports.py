import importlib.util


def test_import(package_name):
    try:
        importlib.util.find_spec(package_name)
        print(f"Import of {package_name} successful!")
    except ImportError:
        print(f"Failed to import {package_name}")


if __name__ == "__main__":
    # Specify the name of your package here
    package_name = "prnn.utils.predictiveNet"

    # Test the import
    test_import("prnn.utils.predictiveNet")
    test_import("prnn.analysis.trajectoryAnalysis")
