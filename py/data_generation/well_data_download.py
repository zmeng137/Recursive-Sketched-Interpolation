from the_well.utils.download import well_download

base_path = "/mnt/CROSS/Well"  

well_download(base_path=base_path, 
              dataset="MHD_256", 
              split="test")