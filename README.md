# ComfyUI-VideoConcat

A simple [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom node for concatenating two videos (image batches) directly. 

## Features
- **Direct Concatenation**: Appends the frames of the second video directly after the first video.
- **Auto Resize**: If the two videos have different resolutions, it automatically resizes the second video to match the first video's resolution using bilinear interpolation to prevent tensor dimension errors.

## Installation

### For Cloud Environments / Manual Installation
1. Navigate to your ComfyUI `custom_nodes` directory.
2. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ComfyUI-VideoConcat.git
   ```
3. Restart ComfyUI.

## Usage
1. Search for the node: **Video Concat (Direct)** (under `Video/Utils` category).
2. Connect the first video (IMAGE batch) to `video1`.
3. Connect the second video (IMAGE batch) to `video2`.
4. Connect the output `video` to a Video Combine or Save Image node.
