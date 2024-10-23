from typing import List, Tuple
def Additional_text(text_input: str, path_image: str, path_pretrained: str, k: int) -> List[str]:
    """
    Find a list of image paths based on additional text and an input image.

    Args:
    - text_input: Additional text to refine the search.
    - path_image: Path to the input image.
    - path_pretrained: path to the pretrained.pt [(Name_image, Torch[...])] 
        - link: https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/23120150_student_hcmus_edu_vn/ETLC51eEv4lJguY0K7dFJvMBl40a9X2kklSyMJlGZt8VzQ?e=AKJbn1
    - k: Number of results to return.

    Returns:
    - result: List of image paths matching the query.
        - Ex: ["../image1.jpg", "../image2.jpg"]
    """

# https://github.com/ABaldrati/CLIP4Cir?tab=readme-ov-file