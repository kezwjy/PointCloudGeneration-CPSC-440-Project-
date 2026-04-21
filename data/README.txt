Raw chair mesh files contained in chairs. Point clouds are created by sampling from these.

chairs/
    /train
        /chair_0001.off
        /chair_0002.off
        .
        .
        .
        /chair_0889.off

    /test
        /chair_0890.off
        /chair_0891.off
        .
        .
        .
        /chair_0989.off

Processed point cloud datasets for each chair are contained in chairs_processed.

./data/chairs_processed/
    /train
        /chair_0001
            /full.npy
            /partial1.npy
            /partial2.npy
        /chair_0002
            /full.npy
            /partial1.npy
            /partial2.npy
        .
        .
        .
        /chair_0989
            /full.npy
            /partial1.npy
            /partial2.npy
        
    /test
        /chair_0890
            /full.npy
            /partial1.npy
            /partial2.npy
        /chair_0891
            /full.npy
            /partial1.npy
            /partial2.npy
        .
        .
        .
        /chair_0989
            /full.npy
            /partial1.npy
            /partial2.npy

For each mesh:
    1. 3000 points are sampled
    2. the following features are extracted: surface variation, local normal alignment, and local point density
    3. the following datasets are created
        1. full.npy: a numpy array containing all the xyz points
            size: (num_points, 3)
            each row contains: [x, y, z]
        2. partial1.npy: a partial point cloud created by randomly sampling
            size: (1000, 6)
            each row contains: [x, y, z, surface variation, alignment, density]
        3. partial2.npy: a partial point cloud created by removing points within a sphere
            size: (1000, 6)
            each row contains: [x, y, z, surface variation, alignment, density]

