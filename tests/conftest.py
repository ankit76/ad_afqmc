def pytest_addoption(parser):
    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help="Run all tests from test_general"
    )
    parser.addoption(
        "--mpi",
        action="store_true",
        default=False,
        help="Run test_general with mpi"
    )
