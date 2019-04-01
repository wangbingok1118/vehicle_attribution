1 : train brand model (all layer)
2 : train attribution model ( all layer)
3 : train brand model ( freeze cov1 cov2 cov3 ) (cov4 cov5 just for brand)
4 : train attribution model (freeze cov1 conv2 conv3) (cov4 conv5 just for attribution)
5 : fusion stage 3 brand model and stage 4 attribution model
6 : add vehicle feature ( don't need train)