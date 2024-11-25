function ROI = extractROI(v, options)

arguments (Input)
    v webcam
    options.ROI (1,4) = [100, 100, 500, 500]
end

snap = snapshot(v);
closePreview(v)

f = figure;
ax = gca;
title("Select your ROI and double click when finished:")
image(ax, snap)
axis(ax, 'off', 'image', 'equal')
colormap gray

roi = drawrectangle(ax, 'Position', options.ROI);
ROI = round(customWait(roi));

snap = rgb2gray(snapshot(v));
snap = imcrop(snap, ROI);

imagesc(ax, snap)
axis(ax, 'off', 'image', 'equal')
colormap gray

dlgTitle    = 'User Question';
dlgQuestion = 'Do like your ROI?';
choice = questdlg(dlgQuestion,dlgTitle,'Yes','No', 'Yes');
pause(0.5)

switch choice

    case 'Yes'
        disp('ROI Extracted...')
        close(f)
        return
    case 'No'
        close(f)
        extract_ROI(v)
end

end

function pos = customWait(hROI)

% Listen for mouse clicks on the ROI
l = addlistener(hROI,'ROIClicked',@clickCallback);

% Block program execution
uiwait;

% Remove listener
delete(l);

% Return the current position
pos = hROI.Position;

end

function clickCallback(~,evt)

if strcmp(evt.SelectionType,'double')
    uiresume;
end

end


