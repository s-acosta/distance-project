function out = make_prepost_gray_videos(varargin)
% MAKE_PREPOST_GRAY_VIDEOS  Write gray PRE and POST videos of desired duration.
% Creates:
%   <basename>_pre.mp4
%   <basename>_post.mp4
% Optional per-frame PNGs and RGB565 .bin frames.
%
% Example:
% out = make_prepost_gray_videos('size',[240 240],'fps',60, ...
%   'pre_seconds',1.0,'post_seconds',1.0, ...
%   'outdir','stim_out_seg','basename','drift', ...
%   'write_png_frames',false,'write_rgb565_frames',true);

p = inputParser;
addParameter(p,'size',[240 240]);       % [H W]
addParameter(p,'fps',20);               % frames/sec
addParameter(p,'pre_seconds',1.0);
addParameter(p,'post_seconds',1.0);
addParameter(p,'codec','MPEG-4');       % or 'Motion JPEG AVI'
addParameter(p,'quality',95);
addParameter(p,'outdir','C:\Users\acostamendoza\projects\distance-project\hardware-control\disparity-device\stimulus');
addParameter(p,'basename','drift');
addParameter(p,'write_png_frames',false);
addParameter(p,'write_rgb565_frames',true);
parse(p,varargin{:});
opt = p.Results;

H = opt.size(1); W = opt.size(2);
nPre  = max(0, round(opt.pre_seconds  * opt.fps));
nPost = max(0, round(opt.post_seconds * opt.fps));

if ~exist(opt.outdir,'dir'), mkdir(opt.outdir); end

% Mean gray frame (uint8)
base = 0.5;
gray = uint8(round(base * 255)) * ones(H, W, 'uint8');
grayRGB = repmat(gray, [1 1 3]);

% ---------- PRE ----------
preTag   = sprintf('%s_pre',  opt.basename);
preVideo = fullfile(opt.outdir, sprintf('%s.mp4', preTag));
prePNG   = "";
preBIN   = "";

if opt.write_png_frames
    prePNG = fullfile(opt.outdir, sprintf('%s_frames_png', preTag));
    if ~exist(prePNG,'dir'), mkdir(prePNG); end
end
if opt.write_rgb565_frames
    preBIN = fullfile(opt.outdir, sprintf('%s_frames_rgb565', preTag));
    if ~exist(preBIN,'dir'), mkdir(preBIN); end
    write_readme(preBIN, W, H);
end

writer = make_writer(preVideo, opt.codec, opt.quality, opt.fps);

% PRE loop
for i = 1:nPre
    frame = grayRGB;
    writeVideo(writer, frame);
    if ~isempty(prePNG)
        imwrite(frame, fullfile(prePNG,  sprintf('frame_%06d.png', i)));
    end
    if ~isempty(preBIN)
        payload = rgb565_payload(frame);
        fbin = fullfile(preBIN, sprintf('frame_%06d.bin', i));
        fid = fopen(fbin,'w'); fwrite(fid, payload, 'uint8'); fclose(fid);
    end
end
close(writer);

% ---------- POST ----------
postTag   = sprintf('%s_post', opt.basename);
postVideo = fullfile(opt.outdir, sprintf('%s.mp4', postTag));
postPNG   = "";
postBIN   = "";

if opt.write_png_frames
    postPNG = fullfile(opt.outdir, sprintf('%s_frames_png', postTag));
    if ~exist(postPNG,'dir'), mkdir(postPNG); end
end
if opt.write_rgb565_frames
    postBIN = fullfile(opt.outdir, sprintf('%s_frames_rgb565', postTag));
    if ~exist(postBIN,'dir'), mkdir(postBIN); end
    write_readme(postBIN, W, H);
end

writer = make_writer(postVideo, opt.codec, opt.quality, opt.fps);

% POST loop
for i = 1:nPost
    frame = grayRGB;
    writeVideo(writer, frame);
    if ~isempty(postPNG)
        imwrite(frame, fullfile(postPNG, sprintf('frame_%06d.png', i)));
    end
    if ~isempty(postBIN)
        payload = rgb565_payload(frame);
        fbin = fullfile(postBIN, sprintf('frame_%06d.bin', i));
        fid = fopen(fbin,'w'); fwrite(fid, payload, 'uint8'); fclose(fid);
    end
end
close(writer);

out = struct('pre_video',preVideo,'post_video',postVideo, ...
             'pre_png_dir',prePNG,'post_png_dir',postPNG, ...
             'pre_rgb565_dir',preBIN,'post_rgb565_dir',postBIN, ...
             'fps',opt.fps,'size',[H W],'nPre',nPre,'nPost',nPost);
end

% ---------- helpers ----------
function vw = make_writer(path, codec, quality, fps)
    vw = VideoWriter(path, codec);
    if strcmpi(codec,'MPEG-4')
        vw.Quality = quality;
    end
    vw.FrameRate = fps;
    open(vw);
end

function write_readme(dirpath, W, H)
    fp = fullfile(dirpath,'README.txt');
    if exist(fp,'file'), return; end
    fid = fopen(fp,'w');
    fprintf(fid,'RGB565 frames, row-major LITTLE-endian, size %dx%d.\n',W,H);
    fprintf(fid,'Send header "IMG %d %d\\n" then payload bytes (W*H*2).\n',W,H);
    fclose(fid);
end

function payload = rgb565_payload(imgRGB)
    if ~isa(imgRGB,'uint8'), imgRGB = im2uint8(imgRGB); end
    R5 = bitshift(uint16(imgRGB(:,:,1)),-3);
    G6 = bitshift(uint16(imgRGB(:,:,2)),-2);
    B5 = bitshift(uint16(imgRGB(:,:,3)),-3);
    rgb565 = bitor(bitshift(R5,11), bitor(bitshift(G6,5), B5));   % HxW uint16
    payload = typecast(reshape(rgb565.',[],1),'uint8');           % row-major LE bytes
end
