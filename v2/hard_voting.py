import argparse
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import glob


def voting(file_dir, csv1, csv2, csv3, save_dir):
    """
    csv파일 3개를 받아서 투표를 진행하는 함수
    
    Args:
        file_dir (str): csv 파일이 있는 디렉토리 경로
        csv1 (str): csv 파일 이름
        csv2 (str): csv 파일 이름
        csv3 (str): csv 파일 이름
        save_dir (str): 저장할 디렉토리 경로
    
    Returns:
        save_csv (str): 저장할 csv 파일 이름
    """    
    load_1 = pd.read_csv(os.path.join(file_dir, csv1))
    load_2 = pd.read_csv(os.path.join(file_dir, csv2))
    load_3 = pd.read_csv(os.path.join(file_dir, csv3))
    voting_result = []

    for value1, value2, value3 in zip(load_1['ans'], load_2['ans'], load_3['ans']):
        vote_list = [0] * 18
        vote_list[value1] += 1
        vote_list[value2] += 1
        vote_list[value3] += 1
        if vote_list.count(max(vote_list)) == 1:
            voting_result.append(vote_list.index(max(vote_list)))
        else:
            voting_result.append(value1)

    load_1['ans'] = voting_result
    KST = timezone(timedelta(hours=9))
    current_time = datetime.now(KST).strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_dir, f"{current_time}.csv")
    load_1.to_csv(save_path, index=False)
    print(f"voting result save path: {save_path}")


if __name__ == "__main__":
    """
    file_dir : csv 경로
    csv1 : csv 파일 이름(1등)
    csv2 : csv 파일 이름(2등)
    csv3 : csv 파일 이름(3등)
    save_dir : 저장할 경로
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv1",
        type=str,
        default="csv1.csv",
        help="csv type (default: csv1)"
    )
    parser.add_argument(
        "--csv2",
        type=str,
        default="csv2.csv",
        help="csv type (default: csv2)"
    )
    parser.add_argument(
        "--csv3",
        type=str,
        default="csv3.csv",
        help="csv type (default: csv3)"
    )
    parser.add_argument(
        "--file_dir",
        type=str,
        default="./output",
        help="file type (default: ./output)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./voting",
        help="save type (default: ./voting)"
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    voting(args.file_dir, args.csv1, args.csv2, args.csv3, args.save_dir)