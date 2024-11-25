%% This script is to preview the images to fine tune the position of the cameras

function previewCamera(v, options)

arguments (Input)
    v
    options.Flip (1,1) logical = false
end

fig = figure('Units','normalized', 'Position', [0.1,0.2,0.4,0.5]);

ax1 = gca;

if options.Flip
    im1 = image(ax1, flipud(snapshot(v)));
else
     im1 = image(ax1, snapshot(v));
end

title(ax1, 'Mouse Camera', 'FontSize', 14);
axis(ax1, 'off', 'image', 'equal')

i_frame = 0;
while true

    i_frame = i_frame + 1;

    snap_1 = snapshot(v);

    if ishandle(fig)

        if options.Flip
            set(im1, 'CData', flipud(snap_1))
        else
            set(im1, 'CData', snap_1)
        end
        
        sgtitle(fig, "Frame " + string(i_frame))

    else
        break
    end

end

end









