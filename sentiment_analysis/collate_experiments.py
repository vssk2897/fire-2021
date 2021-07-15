from .constants import C
import os
import re
import pandas

class collate_experiments:
    
    def __init__(self):
        self.pattern = r'.*(eval.tsv)'

    def get_results(self):
        df = pandas.DataFrame()
        base_path = os.path.join(os.getcwd(), C.EXPERIMENT_DIR)
        for root, dirs, files in os.walk(base_path):
            if len(files) != 0:
                for file in files:
                    match = re.match(self.pattern, file)
                    if match:
                        path = os.path.join(root, file)
                        if df.empty:
                            df = pandas.read_csv(path, sep=C.SEPERATOR)
                            
                        else :
                            tdf = pandas.read_csv(path, sep=C.SEPERATOR)
                            df = pandas.concat([df, tdf], ignore_index=True)
                           

        return df