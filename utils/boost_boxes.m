function [sa] = boost_boxes(ba, sa, bb, sb)

nb = size(ba,1); 
iou_thr = 0.3;

box_a = [ba(:,1:2) ba(:,3:4)-ba(:,1:2)+1];
box_b =    [bb(:,1:2) bb(:,3:4)-bb(:,1:2)+1];


for i=1:nb
    ovlp = inters_union(box_a(i,:), box_b); % ovlp has 1x5 or 5x1 dim
    [movlp, mind] = max(ovlp);
    if movlp>=iou_thr;
        sa(i,:) = sa(i,:) + sb(mind,:)*movlp;
%         bar([sa(i,:); sb(mind,:)]);
    end
end

end

% ------------------------------------------------------------------------
function iou = inters_union(bounds1,bounds2)
% ------------------------------------------------------------------------
inters = rectint(bounds1,bounds2);
ar1 = bounds1(:,3).*bounds1(:,4);
ar2 = bounds2(:,3).*bounds2(:,4);
union = bsxfun(@plus,ar1,ar2')-inters;
iou = inters./(union+0.001);
end
