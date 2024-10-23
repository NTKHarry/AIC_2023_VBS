from typing import List, Tuple
def FindNQuery(list_query: List[str], path_pretrained: str, K: int, threshold: int) -> List[Tuple[List[str], int, int, int]]:
    """
    Find images matching the queries and return a list of tuples containing image paths, video name, frame start, and frame end.

    Args:
    - list_query: List of query strings to search for in the images.
    - path_pretrained: path to file pretrained
      - link: https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/23120150_student_hcmus_edu_vn/ETLC51eEv4lJguY0K7dFJvMBl40a9X2kklSyMJlGZt8VzQ?e=AKJbn1

    - K: Number of results to return for each query.
    - threshold: Maximum allowed distance between images for similarity.

    Returns:
    - result: List of tuples, where each tuple contains:
      - A list of image paths that match the query.
      - The name of the video containing the images. (Ex: Video_100_23, frame 23-th in video 100-th) 
      - The start frame of the video.
      - The end frame of the video.
    """