

# cls 별로 loss와 acc 출력하기
# 함수 input으로 받아올거 (labels, loss 값, preds, outs, criterion, args.batchsize, args.log_interval)



class AccuracyLoss():
    #loss = criterion(outs, labels)
    #loss_value += loss.item()
    #matches += (preds == labels).sum().item()
    def __init__(self, labels, preds, outs, criterion):
        # criterion = create_criterion(criterion)  # default: cross_entropy
        mask_wear_loss_value = 0 
        mask_wear_matches = 0
        mask_incorrect_loss_value = 0
        mask_incorrect_matches = 0
        mask_not_wear_loss_value = 0
        mask_not_wear_matches = 0
       
        male_loss_value = 0
        male_matches = 0
        female_loss_value = 0
        female_matches = 0
        
        age_0_30_loss_value = 0
        age_0_30_matches = 0
        age_30_60_loss_value = 0
        age_30_60_matches = 0
        age_60_loss_value = 0
        age_60_loss_matches = 0

        self.wear_cnt = 0
        self.incorrect_cnt = 0
        self.not_wear_cnt = 0

        self.male_cnt = 0
        self.female_cnt = 0
        
        self.age_0_30_cnt = 0
        self.age_30_60_cnt = 0
        self.age_60_cnt = 0

        for label, pred, out in zip(labels, preds, outs):
            # Mask
            if label in [0, 1, 2, 3, 4, 5]: # Wear
                mask_wear_loss = criterion(out, label)
                mask_wear_loss_value += mask_wear_loss.item()
                mask_wear_matches += (pred == label)
                self.wear_cnt += 1
            elif label in [6, 7, 8, 9, 10, 11]: # Incorrect
                mask_incorrect_loss = criterion(out, label)
                mask_incorrect_loss_value += mask_incorrect_loss.item()
                mask_incorrect_matches += (pred == label)
                self.incorrect_cnt += 1
            elif label in [12, 13, 14, 15, 16, 17]: # Not Wear
                mask_not_wear_loss = criterion(out, label)
                mask_not_wear_loss_value += mask_not_wear_loss.item()
                mask_not_wear_matches += (pred == label)
                self.not_wear_cnt += 1

            # Gender
            if label in [0, 1, 2, 6, 7, 8, 12, 13, 14]: # male
                male_loss = criterion(out, label)
                male_loss_value += male_loss.item()
                male_matches += (pred == label)
                self.male_cnt += 1
            else: # female
                female_loss = criterion(out, label)
                female_loss_value += female_loss.item()
                female_matches += (pred == label)
                self.female_cnt += 1
                
            # Age
            if label % 3 == 0: # 30 이하
                age_0_30_loss = criterion(out, label)
                age_0_30_loss_value += age_0_30_loss.item()
                age_0_30_matches += (pred == label)
                self.age_0_30_cnt += 1
            elif label % 3 == 1: # 30~60
                age_30_60_loss = criterion(out, label)
                age_30_60_loss_value += age_30_60_loss.item()
                age_30_60_matches += (pred == label)
                self.age_30_60_cnt += 1
            else: # 60 이상
                age_60_loss = criterion(out, label)
                age_60_loss_value += age_60_loss.item()
                age_60_loss_matches += (pred == label)
                self.age_60_cnt += 1

        self.loss_dict = {
            'mask_wear_loss_value' : mask_wear_loss_value,
            'mask_incorrect_loss_value' : mask_incorrect_loss_value,
            'mask_not_wear_loss_value' : mask_not_wear_loss_value,

            'male_loss_value' : male_loss_value,
            'female_loss_value' : female_loss_value,
            
            'age_0_30_loss_value' : age_0_30_loss_value,
            'age_30_60_loss_value' : age_30_60_loss_value,
            'age_60_loss_value' : age_60_loss_value,
        }
        self.matches_dict = {
            'mask_wear_matches' : mask_wear_matches,
            'mask_incorrect_matches' : mask_incorrect_matches,
            'mask_not_wear_matches' : mask_not_wear_matches,

            'male_matches' : male_matches,
            'female_matches' : female_matches,

            'age_0_30_matches' : age_0_30_matches,
            'age_30_60_loss_matches' : age_30_60_matches,
            'age_60_loss_matches' : age_60_loss_matches
        }

    def loss_acc(self, iter, len_set):

        # train_loss = loss_value / args.log_interval
        # train_acc = matches / args.batch_size / args.log_interval

        mask_wear_loss = self.loss_dict['mask_wear_loss_value'] / iter
        mask_incorrect_loss = self.loss_dict['mask_incorrect_loss_value'] / iter
        mask_not_wear_loss = self.loss_dict['mask_not_wear_loss_value'] / iter
        male_loss = self.loss_dict['male_loss_value'] / iter
        female_loss = self.loss_dict['female_loss_value'] / iter
        age_0_30_loss = self.loss_dict['age_0_30_loss_value'] / iter
        age_30_60_loss = self.loss_dict['age_30_60_loss_value'] / iter
        age_60_loss = self.loss_dict['age_60_loss_value'] / iter

        mask_wear_acc = self.matches_dict['mask_wear_matches'] / self.wear_cnt  if self.wear_cnt!=0 else 0
        mask_incorrect_acc = self.matches_dict['mask_incorrect_matches'] / self.incorrect_cnt if self.incorrect_cnt!=0 else 0
        mask_not_wear_acc = self.matches_dict['mask_not_wear_matches'] / self.not_wear_cnt if self.not_wear_cnt!=0 else 0
        male_acc = self.matches_dict['male_matches'] / self.male_cnt  if self.male_cnt!=0 else 0
        female_acc = self.matches_dict['female_matches'] / self.female_cnt if self.female_cnt!=0 else 0
        age_0_30_acc = self.matches_dict['age_0_30_matches'] / self.age_0_30_cnt if self.age_0_30_cnt!=0 else 0
        age_30_60_acc = self.matches_dict['age_30_60_loss_matches'] / self.age_30_60_cnt if self.age_30_60_cnt!=0 else 0
        age_60_acc = self.matches_dict['age_60_loss_matches'] / self.age_60_cnt if self.age_60_cnt!=0 else 0

        loss_dict = {
            'mask_wear_loss' : mask_wear_loss,
            'mask_incorrect_loss' : mask_incorrect_loss,
            'mask_not_wear_loss' : mask_not_wear_loss,
            'male_loss' : male_loss,
            'female_loss' : female_loss,
            'age_0_30_loss' : age_0_30_loss,
            'age_30_60_loss' : age_30_60_loss,
            'age_60_loss' : age_60_loss,
        }
        acc_dict = {
            'mask_wear_acc' : mask_wear_acc / len_set,
            'mask_incorrect_acc' : mask_incorrect_acc / len_set,
            'mask_not_wear_acc' : mask_not_wear_acc / len_set,
            'male_acc' : male_acc / len_set,
            'female_acc' : female_acc / len_set,
            'age_0_30_acc' : age_0_30_acc / len_set,
            'age_30_60_acc' : age_30_60_acc / len_set,
            'age_60_acc' : age_60_acc / len_set,
        }

        return loss_dict, acc_dict