function overlap_ratio = get_overlap_MtoN(rect1, rect2)
    if isempty(rect1) || isempty(rect2) 
      overlap_ratio = []; return;
    end
    overlap_ratio = zeros(size(rect1, 1), size(rect2, 1));
    for i = 1 : size(rect1, 1)
        overlap_ratio(i,:) = get_overlap_1toN(rect1(i,:), rect2);
    end
end