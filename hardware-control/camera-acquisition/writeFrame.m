function [snap, timestamp] = writeFrame(v, logfile, ROI)

arguments (Input)
    v webcam
    logfile VideoWriter
    ROI (1,4) 
end

[snap, timestamp] = snapshot(v);
snap = rgb2gray(snap);
snap = imcrop(snap, ROI);
writeVideo(logfile, snap);

end