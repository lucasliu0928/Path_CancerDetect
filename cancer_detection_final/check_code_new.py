def predict(net, data_loader, device, conf, header):    
    y_pred = []
    y_true = []
    y_pred_prob = []
    # Set the network to evaluation mode
    net.eval()
    for data in data_loader:
        image_patches = data[0].to(device, dtype=torch.float32)
        label_lists = data[1][0]
        sub_preds_list, slide_preds_list, attn_list = net(image_patches) #lists len of n of tasks, each task = [5,2], [1,2], [1,5,3],
        
        #Compute loss for each task, then sum
        pred_list = []
        pred_prob_list = []
        for k in range(conf.n_task):
            sub_preds = sub_preds_list[k]
            slide_preds = slide_preds_list[k]
            attn = attn_list[k]
            labels = label_lists[:,k].to(device, dtype = torch.float32).to(device)
            pred_prob = torch.sigmoid(slide_preds)
            pred = pred_prob[0][0].round()
            pred_list.append(pred)
            pred_prob_list.append(pred_prob)
    
        y_pred.append(pred_list)
        y_true.append(label_lists)
        y_pred_prob.append(pred_prob_list)

    #Get prediction for each task
    y_predprob_task = []
    y_pred_tasks = []
    y_true_tasks = []
    for k in range(conf.n_task):
        y_pred_tasks.append([p[k] for p in y_pred])
        y_predprob_task.append([p[k].item() for p in y_pred_prob])
        y_true_tasks.append([t[:,k].to(device, dtype = torch.int64).item() for t in y_true])
    
    return y_pred_tasks, y_predprob_task, y_true_tasks