import os
import sys
CONTAINING_FOLDER = os.path.realpath(__file__)

for _ in range(3):
    CONTAINING_FOLDER = os.path.dirname(CONTAINING_FOLDER)
    
sys.path.append(CONTAINING_FOLDER)