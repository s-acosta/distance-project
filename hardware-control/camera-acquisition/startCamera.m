function v = startCamera(camera_id, options)

arguments (Input)
    camera_id (1,1) {mustBeNumeric}
    options.WriteVideo (1,1) logical = false
    options.ROISelection (1,1) logical = true
    options.Resolution (1,1) string = '640x360'
    options.Preview (1,1) logical = false
end

arguments (Output)
    v webcam
end

v = webcam(camera_id);
v.Resolution = options.Resolution;

if options.Preview
    preview(v);
end

end

