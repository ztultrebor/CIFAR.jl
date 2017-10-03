function labelnames()
  # Set File Name
  DataDir = joinpath(dirname(@__FILE__),"..","data","bin")
  LabelNameFile = joinpath(DataDir,"batches.meta.txt")

  # Open file
  h_LabelNameFile = open(LabelNameFile)

  # Read Labels
  LabelNames = readlines(h_LabelNameFile)

  # Close File
  close(h_LabelNameFile)
  return LabelNames[1:end-1]
end


function traindata(N=50000)
    Pixels = 1024
    Labels = 10

    # Fixed dataset values
    Features = 3*Pixels
    Samples = 10000
    Batches = 5

    X = Array{Float32,2}(Features,Batches*Samples)
    Y = Array{Float32,2}(Labels,Batches*Samples)

    for batch_number = 1:Batches
        # Assert to the number of batches
        DataDir = joinpath(dirname(@__FILE__),"..","data","bin")
        BatchFile = @sprintf "data_batch_%d.bin" batch_number
        BatchFile = joinpath(DataDir,BatchFile)
        LabelNameFile = joinpath(DataDir,"batches.meta.txt")

        # Open Batch File
        h_BatchFile = open(BatchFile)

        # Loop Over Samples
        r = Array{UInt8,1}(Pixels)
        g = Array{UInt8,1}(Pixels)
        b = Array{UInt8,1}(Pixels)
        for j = 1:Samples
            # Read Label Byte
            idx = (batch_number-1)*Samples + j
            tgt = read(h_BatchFile,UInt8)+1
            Y[:,idx] = [tgt == i for i = 1:Labels]
            # Read Color Byte Arrays
            read!(h_BatchFile,r)
            read!(h_BatchFile,g)
            read!(h_BatchFile,b)

            X[1:Pixels,idx] = r/256
            X[Pixels+1:2*Pixels,idx] = g/256
            X[2*Pixels+1:end,idx] = b/256
        end

        # Close File
        close(h_BatchFile)

    end
    return X[:,1:N],Y[:,1:N]
end


function testdata(N=10000)
    Pixels = 1024
    Labels = 10

    # Fixed dataset values
    Features = 3*Pixels
    Samples = 10000

    X = Array{Float32,2}(Features,Samples)
    Y = Array{Float32,2}(Labels,Samples)

    DataDir = joinpath(dirname(@__FILE__),"..","data","bin")
    BatchFile = "test_batch.bin"
    BatchFile = joinpath(DataDir,BatchFile)

    h_BatchFile = open(BatchFile)

    # Loop Over Samples
    r = Array{UInt8,1}(Pixels)
    g = Array{UInt8,1}(Pixels)
    b = Array{UInt8,1}(Pixels)
    for idx = 1:Samples
        # Read Label Byte
        tgt = read(h_BatchFile,UInt8)+1
        Y[:,idx] = [tgt == i for i = 1:Labels]
        # Read Color Byte Arrays
        read!(h_BatchFile,r)
        read!(h_BatchFile,g)
        read!(h_BatchFile,b)

        X[1:Pixels,idx] = r/256
        X[Pixels+1:2*Pixels,idx] = g/256
        X[2*Pixels+1:end,idx] = b/256
    end

    # Close File
    close(h_BatchFile)
    return X[:,1:N],Y[:,1:N]
end
