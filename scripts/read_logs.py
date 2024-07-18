import numpy as np
import random
import imageio
from IPython.display import HTML

def read_arrays_from_file(obs_filename, title_filename, compute_sample_reward_filename, compute_main_reward_filename):
    obs = []
    titles = []
    csr = []
    cmr = []

    # Read titles
    with open(title_filename, 'r') as f:
        titles = f.read().strip().split('\n')

    # Read arrays
    with open(obs_filename, 'rb') as f:
        while True:
            try:
                array = np.load(f)
                obs.append(array)
            except ValueError:
                break
    
    with open(compute_main_reward_filename, 'rb') as f:
        while True:
            try:
                array = np.load(f)
                cmr.append(array)
            except ValueError:
                break

    if compute_sample_reward_filename is not "":                
        with open(compute_sample_reward_filename, 'rb') as f:
            while True:
                try:
                    array = np.load(f)
                    csr.append(array)
                except ValueError:
                    break

    return titles, obs, csr, cmr

def obs_to_video(images, filename):
    """
    converts a list of images to video and writes the file
    """
    video_writer = imageio.get_writer(filename, fps=60)
    for image in images:
        video_writer.append_data(image[::-1])
    video_writer.close()
    HTML("""
        <video width="640" height="480" controls>
            <source src="output.mp4" type="video/mp4">
        </video>
        <script>
            var video = document.getElementsByTagName('video')[0];
            video.playbackRate = 2.0; // Increase the playback speed to 2x
            </script>    
    """)

if __name__ == "__main__":
    obs_path = r"/Users/faraz/Documents/College/Texas/Thesis/Curriculum-Learning-for-Visual-Manipulation/20240626-014431_obs.npy"
    title_path = r"/Users/faraz/Documents/College/Texas/Thesis/Curriculum-Learning-for-Visual-Manipulation/20240626-014431_episode_count.txt"
    cmr_path = r"/Users/faraz/Documents/College/Texas/Thesis/Curriculum-Learning-for-Visual-Manipulation/20240626-014431_compute_main_reward.npy"
    csr_path = ""
    titles, obs, csr, cmr = read_arrays_from_file(obs_path, title_path, csr_path, cmr_path)

    # generate random video
    max_length = 250
    start = random.randint(0, len(obs) - 1)
    end = start + max_length if (start + max_length < len(obs) - 1) else len(obs) - 1
    print(start, end)
    print(f"generating video from {titles[start]} to {titles[end]}")
    images = []
    for i in range(start, end):
        images.append(obs[i])

    obs_to_video(images, "random_video.mp4")
    

