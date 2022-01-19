import os
import pandas as pd


def join_data():
    write_path = lambda x: f'./Data/{x}.csv'
    filepaths = [os.path.splitext(f)[0] for f in os.listdir('./Data') if
                 os.path.isfile(write_path(os.path.splitext(f)[0]))]

    df = df_aux = pd.DataFrame()
    for i in filepaths:
        df_aux = pd.read_csv(write_path(i))
        df_aux['Class'] = i

        df = df.append(df_aux)

    df.to_csv(write_path('full'), index=False)

    return None


if __name__ == '__main__':
    join_data()
