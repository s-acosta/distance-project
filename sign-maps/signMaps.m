% reference =  reference_3;

% load('SAM017_231016_additional_maps.mat');
reference = snap;
reference = imresize(reference, size(maps.VFS_processed));
maps.HorizontalRetinotopy = maps.HorizontalRetinotopy + 45;

%%
field_of_view_vertical = maps.VerticalRetinotopy > -5 & maps.VerticalRetinotopy < 5;
field_of_view_horizontal = maps.HorizontalRetinotopy > 0;
field_of_view_V1 = maps.VFS_processed < 0;

field_of_view = field_of_view_vertical & field_of_view_horizontal & field_of_view_V1;

%% Plot VFS
figure(1);
ax1 = axes; 
im = imagesc(ax1, reference); 
im.AlphaData = 1; % change this value to change the background image transparency 
axis equal; 
hold all; 
%plot second data 
ax2 = axes; 
im1 = imagesc(ax2, imgaussfilt(maps.VFS_raw, 1.5)); 
im1.AlphaData = 0.3; % change this value to change the foreground image transparency 
axis equal; 
%%Hide the top axes 
ax2.Visible = 'off'; 
ax2.XTick = []; 
ax2.YTick = []; 
ax1.Visible = 'off'; 
ax1.XTick = []; 
ax1.YTick = []; 
linkaxes([ax1,ax2]) 
%add differenct colormap to different data if you wish 
colormap(ax1,'gray') 
colormap(ax2,'jet') 
%set the axes and colorbar position 
set([ax1,ax2],'Position',[.17 .11 .685 .815]); 
% colorbar(ax2)
title('VFS', fontsize = 16)

%% Plot Vertical
figure(2);
ax1 = axes; 
im = imagesc(ax1, snap); 
im.AlphaData = 1; % change this value to change the background image transparency 
axis square; 
hold all; 
%plot second data 
ax2 = axes; 
im1 = imagesc(ax2, alt); 
im1.AlphaData = 0.5; % change this value to change the foreground image transparency 
axis square; 
%link axes 
linkaxes([ax1,ax2]) 
%%Hide the top axes 
ax2.Visible = 'off'; 
ax2.XTick = []; 
ax2.YTick = []; 
ax1.Visible = 'off'; 
ax1.XTick = []; 
ax1.YTick = []; 
%add differenct colormap to different data if you wish 
colormap(ax1,'gray') 
colormap(ax2,'jet') 
clim([-60, 60])
colorbar
%set the axes and colorbar position 
set([ax1,ax2],'Position',[.17 .11 .685 .815]); 
%% Plot Horizontal
figure(3);
ax1 = axes; 
im = imagesc(ax1, snap); 
im.AlphaData = 1; % change this value to change the background image transparency 
axis square; 
hold all; 
%plot second data 
ax2 = axes; 
im1 = imagesc(ax2, azi); 
im1.AlphaData = 0.5; % change this value to change the foreground image transparency 
axis square; 
%link axes 
linkaxes([ax1,ax2]) 
%%Hide the top axes 
ax2.Visible = 'off'; 
ax2.XTick = []; 
ax2.YTick = []; 
ax1.Visible = 'off'; 
ax1.XTick = []; 
ax1.YTick = []; 
%add differenct colormap to different data if you wish 
colormap(ax1,'gray') 
colormap(ax2,'jet') 
% clim([0, 90])
colorbar
%set the axes and colorbar position 
set([ax1,ax2],'Position',[.17 .11 .685 .815]); 

%%
%% Plot Horizontal
fig = figure;

ax1 = axes('Parent',fig);
ax2 = axes('Parent',fig);

set(ax1,'Visible','off');
set(ax2,'Visible','off');

imshow(reference, 'Parent', ax1)
hold all
im = imshow(imgaussfilt(maps.VFS_processed, 1.5), 'Parent', ax2, colormap = jet);
set(im,'AlphaData',0.1);
contour(maps.VerticalRetinotopy, [0 0], 'ShowText','on' , 'LineWidth', 2.5, color = 'blue')
contour(maps.HorizontalRetinotopy, [45 45], 'ShowText','on' ,'LineWidth', 2.5, color = 'red')

% hold all
% ax3 = axes;
% 
% hold all
% ax4 = axes;
% 
% 
% linkaxes([ax1, ax2, ax3, ax4]) 

% hold on
% imagesc(reference)



