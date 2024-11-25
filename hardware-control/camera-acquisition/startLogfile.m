function logfile = startLogfile(options)

arguments (Input)
    options.FileLocation (1,1) string = "C:\Users\acostamendoza\Desktop"
    options.FileName (1,1) string = 'test'
    options.FrameRate (1,1) = 10
    options.Quality (1,1) = 100
end

arguments (Output)
    logfile VideoWriter
end

full_filename = fullfile(options.FileLocation, options.FileName);
logfile = VideoWriter(full_filename, "MPEG-4");
logfile.FrameRate = options.FrameRate;
logfile.Quality = options.Quality;

end

