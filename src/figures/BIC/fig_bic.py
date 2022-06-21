from figures import *

sample, scale = commandline_args({"sample":"<required>", "scale":4})

z_mid = 3100
x_mid = 1.875*3456/2

show_section(sample,filename=f"{hdf5_root}/processed/output/figures/{sample}-full-xy-{scale}x.png",scale=scale,axesnames=('x','y'),bbox=(z_mid,(0,-1),(0,-1)),figsize=(10,10))
show_section(sample,filename=f"{hdf5_root}/processed/output/figures/{sample}-full-xz-{scale}x.png",scale=scale,axesnames=('x','z'),bbox=(z_mid,(0,-1),(0,-1)),figsize=(10,10))
show_section(sample,filename=f"{hdf5_root}/processed/output/figures/{sample}-full-yz-{scale}x.png",scale=scale,axesnames=('y','z'),bbox=(z_mid,(0,-1),(0,-1)),figsize=(10,10))
show_section(sample,filename=f"{hdf5_root}/processed/output/figures/{sample}-bic-xy-{scale}x.png",scale=scale,axesnames=('x','y'),bbox=(z_mid,(3000,5000),(3000,4000)),figsize=(10,10))
show_section(sample,filename=f"{hdf5_root}/processed/output/figures/{sample}-bic-yz-{scale}x.png",scale=scale,axesnames=('y','z'),bbox=(x_mid,(3500,5000),(1000,2500)),figsize=(10,10))
