from cleanfid import fid

score = fid.compute_fid("fake", "real")
print("FID:", score)