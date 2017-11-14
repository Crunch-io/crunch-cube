'''Utility functions for crunch cube, as well as other modules.'''
import os
import json


def load_fixture(fixtures_directory, filename):
    '''Loads fixtures for CrunchCube integration tests.'''
    with open(os.path.join(fixtures_directory, filename)) as ctx_file:
        fixture = json.load(ctx_file)
    return fixture
