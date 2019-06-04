#!/usr/bin/python3
# Read metadata and data from raw tomograms from ESRF.
# (C) James Avery for the MAXIBONE project, 2018
import numpy as np;
import bohrium as bh;
import numpy.ma as ma;
import sys,re,os;

# TODO: Switch more elegantly between numpy and bohrium

def esrf_edf_metadata(filename):
    meta = {};
    header_length = 1024;            
    with open(filename,"r",encoding="latin-1") as f:
        header = f.read(header_length);
        ls = header.split("\n");
        for l in ls:
            kv = re.split("[=;]",l);
            if(len(kv)>=2):
                meta[kv[0].strip()] = kv[1].strip();
            
        assert meta["ByteOrder"] == "LowByteFirst";

        if(meta["DataType"] == "UnsignedShort"):
            meta["NumpyType"] = np.uint16;
        if(meta["DataType"] == "Float"):
            meta["NumpyType"] = np.float32;            
        
        return meta;
    
def esrf_edf_to_npy(filename):
    meta = esrf_edf_metadata(filename);
    header_length = 1024;        

    with open(filename,"rb") as f:
        f.seek(header_length,os.SEEK_SET);
        data = np.fromfile(file=f,dtype=meta["NumpyType"]);
#        assert data.shape[0]*2 == int(meta["Size"]);

    (nx,ny) = (int(meta["Dim_2"]), int(meta["Dim_1"]));
    data    = ma.masked_array(data,mask=(data==0)).reshape((nx,ny));
    return (meta,data);

# TODO: function that uses seek to only actually read the appropriate region

def esrf_edf_n_to_npy(info,n):
    dirname        = info["dirname"];
    subvolume_name = info["subvolume_name"].format(n);
    return esrf_edf_to_npy(dirname+"/"+subvolume_name);

def esrf_edfrange_to_npy(info,region):
    [[x_start,y_start,z_start],[x_end,y_end,z_end]] = region;
    assert x_end <= int(info["sizex"]) and y_end <= int(info["sizey"]) and z_end <= int(info["sizez"]);

    shape = (z_end-z_start,y_end-y_start,x_end-x_start);
    image = np.zeros(shape,dtype=np.float32);
    for z in range(z_start,z_end):
        if (z % 10 == 0):
            print(z);
        (meta,data) = esrf_edf_n_to_npy(info,z);
        image[z-z_start] = data[y_start:y_end,x_start:x_end];

    return ma.masked_array(image,mask=(image==0));


def esrf_full_tomogram(info):
    (nx,ny,nz) = (int(info['sizex']),int(info['sizey']),int(info['sizez']));
    return esrf_edfrange_to_npy(info,[[0,0,0],[nx-1,ny-1,nz-1]]);

def esrf_edf_to_bh(filename):
    meta = esrf_edf_metadata(filename);
    (nx,ny) = (int(meta["Dim_2"]), int(meta["Dim_1"]));    
    header_length = 1024;        

    with open(filename,"rb") as f:
        f.seek(header_length,os.SEEK_SET);
        data = bh.fromfile(file=f,dtype=meta["NumpyType"]);
#        assert data.shape[0]*2 == int(meta["Size"]);
        return (meta,data.reshape(ny,nx));

def esrf_edf_n_to_bh(info,n):
    dirname        = info["dirname"];
    subvolume_name = info["subvolume_name"].format(n);
    return esrf_edf_to_bh(dirname+"/"+subvolume_name);

def esrf_edfrange_to_bh(info,region):
    [[x_start,y_start,z_start],[x_end,y_end,z_end]] = region;
    assert x_end <= int(info["sizex"]) and y_end <= int(info["sizey"]) and z_end <= int(info["sizez"]);

    shape = (z_end-z_start,y_end-y_start,x_end-x_start);
    image = bh.zeros(shape,dtype=bh.float32);
    for z in range(z_start,z_end):
        if (z % 100 == 0):
            print(z);
        (meta,data) = esrf_edf_n_to_bh(info,z);

        image[z-z_start] = data[y_start:y_end,x_start:x_end];

    return image;

def esrf_full_tomogram_bh(info):
    (nx,ny,nz) = (int(info['sizex']),int(info['sizey']),int(info['sizez']));
    return esrf_edfrange_to_bh(info,[[0,0,0],[nx-1,ny-1,nz-1]]);


def esrf_read_xml(filename):
    fields = ["subvolume_name","sizex","sizey","sizez","originx","originy","originz","voxelsize","valmin","valmax","byte_order","s1","s2","S1","S2"];
    fieldstrings = ["\<{}\>(.*)\<\/{}\>".format(f,f) for f in fields];
    res = [re.compile(s,re.IGNORECASE) for s in fieldstrings];
    xmlmeta = {};    
    with open(filename,"r") as file:
        for l in file.readlines():
            for i in range(len(fields)):
                m = res[i].match(l);
                if(m):
                    xmlmeta[fields[i]] = m.groups()[0];
        xmlmeta["experiment"]=xmlmeta["subvolume_name"].replace("_%04d.edf","");
        xmlmeta["subvolume_name"]=xmlmeta["subvolume_name"].replace("%04d","{:04d}");
        xmlmeta["filename"]=filename;
        xmlmeta["dirname"]=os.path.dirname(filename);
    
    # Change printf template to python3 format template
    return xmlmeta;

        
def frame_histogram(frame,i,bin_edges):
#    print("Calculating histogram for frame",i)        
    count =  np.histogram(frame.compressed(),bins=bin_edges)[0];
#    print("Completed histogram for frame",i)
    return count

#To get a total histogram, simply do np.sum(count,axis=0)
def progressive_histogram(xml,nbins=2048,bin_edges=np.array([]),num_cores=4):
    
    if(len(bin_edges)==0):
        bin_edges = np.linspace(float(xml["valmin"]), float(xml["valmax"]), nbins + 1);
        nbins = len(bin_edges)-1;


    nz     = int(xml["sizez"]);
    print("sizez = ",nz)    
    meta,frame  = esrf_edf_n_to_npy(xml,0);
    frames = np.ma.empty((4*num_cores, frame.shape[0], frame.shape[1]));
    counts = np.empty((nz,nbins),dtype=int);
    
    for i in range(0,nz,4*num_cores):
        chunk_length = min(4*num_cores,nz-i);
        for j in range(chunk_length):
            print("Reading data frame",i+j);
            _, frames[j] = esrf_edf_n_to_npy(xml,i+j);
        counts[i:i+chunk_length] = np.array(Parallel(n_jobs=num_cores)(delayed(frame_histogram)(frames[j],i+j,bin_edges)
                                                                      for j in range(chunk_length)));
        
    return counts, bin_edges;
        
