from ears import EARSSelector

result = EARSSelector().select("video.mp4", k=4, return_frames=True)

print(result.frame_indices)
print(result.timestamps)
print(result.scores)
result.save_frames("selected_frames")
