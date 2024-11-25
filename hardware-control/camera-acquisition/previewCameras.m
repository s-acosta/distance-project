%% This script is to preview the images to fine tune the position of the cameras

function previewCameras(v,w)

fig = figure('Units','normalized', 'Position', [0.1,0.2,0.8,0.5]);

ax1 = subplot(1,2,1);
ax2 = subplot(1,2,2);

im1 = image(ax1, flipud(snapshot(w)));
im2 = image(ax2, flipud(snapshot(v)));

title(ax1, 'Right Eye', 'FontSize', 14);
title(ax2, 'Left Eye', 'FontSize', 14);

axis([ax1, ax2], 'off', 'image', 'equal')

i_frame = 0;
while true

    i_frame = i_frame + 1;

    snap_1 = snapshot(v);
    snap_2 = snapshot(w);

    if ishandle(fig)
        set(im1, 'CData', flipud(snap_1))
        set(im2, 'CData', flipud(snap_2))

        sgtitle(fig, "Frame " + string(i_frame) )
    else
        break
    end

end

end









