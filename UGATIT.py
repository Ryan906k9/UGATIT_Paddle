import time
from networks import *
from utils import *
import numpy as np
import cv2
from glob import glob
import os
import paddle.fluid as fluid
import paddle


class UGATIT(object):
    def __init__(self, args):
        self.light = args.light

        if self.light:
            self.model_name = 'UGATIT_light'
        else:
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        # if torch.backends.cudnn.enabled and self.benchmark_flag:
        #    print('set benchmark !')
        #    torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)




    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """

        ## 这一块代码的主要作用是读取数据，对数据做预处理，然后加载到 DataLoader 中
        ## 所以 paddle 实现中，也可以分块来实现：数据预处理 + paddle.fluid.io.DataLoader

        # train_transform = transforms.Compose([
        #    transforms.RandomHorizontalFlip(),
        #    transforms.Resize((self.img_size + 30, self.img_size+30)),
        #    transforms.RandomCrop(self.img_size),
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        # ])
        # test_transform = transforms.Compose([
        #    transforms.Resize((self.img_size, self.img_size)),
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        # ])



        # self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform)
        # self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), train_transform)
        # self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), test_transform)
        # self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'), test_transform)
        self.trainA = custom_reader(os.path.join(self.dataset, 'trainA'))
        self.trainB = custom_reader(os.path.join(self.dataset, 'trainB'))
        self.testA = custom_reader(os.path.join(self.dataset, 'testA'))
        self.testB = custom_reader(os.path.join(self.dataset, 'testB'))

        # self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True)
        # self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True)
        # self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
        # self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)
        self.trainA_loader = fluid.io.batch(
            paddle.fluid.io.shuffle(self.trainA, buf_size=50000),
            batch_size=self.batch_size)  # 读取训练数据
        self.trainB_loader = fluid.io.batch(
            paddle.fluid.io.shuffle(self.trainB, buf_size=50000),
            batch_size=self.batch_size)  # 读取训练数据
        self.testA_loader = fluid.io.batch(
            paddle.fluid.io.shuffle(self.testA, buf_size=50000),
            batch_size=1)  # 读取训练数据
        self.testB_loader = fluid.io.batch(
            paddle.fluid.io.shuffle(self.testB, buf_size=50000),
            batch_size=1)  # 读取训练数据

        """ Define Generator, Discriminator """
        ## 这里都是 networks 中已经定义好的类进行实例化，仅需修改 .to(self.device) ，这个是在下面 train 函数前定义的

        # self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        # self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        # self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        # self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        # self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
        # self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
        place = fluid.CUDAPlace(0) if self.device == 'cuda' else fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            # 把 networks 中的模型进行实例化
            self.genA2B = ResnetGenerator('genA2B',input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res,
                                          img_size=self.img_size, light=self.light)
            self.genB2A = ResnetGenerator('genB2A',input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res,
                                          img_size=self.img_size, light=self.light)
            self.disGA = Discriminator_G('disGA',input_nc=3, ndf=self.ch, n_layers=7)
            self.disGB = Discriminator_G('disGB',input_nc=3, ndf=self.ch, n_layers=7)
            self.disLA = Discriminator_L('disLA',input_nc=3, ndf=self.ch, n_layers=5)
            self.disLB = Discriminator_L('disLB',input_nc=3, ndf=self.ch, n_layers=5)
            """ Define Loss """
        ## paddle 貌似是在 train 函数中进行定义的？

            # self.L1_loss = nn.L1Loss().to(self.device)
            self.L1_loss = paddle.fluid.dygraph.L1Loss()
            # 下面这两个不需要实例化，直接用？？？
            #self.MSE_loss = nn.MSELoss().to(self.device)
            self.MSE_loss = paddle.fluid.dygraph.MSELoss()
            #self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)
            self.BCE_loss = BCEWithLogitsLoss()

            """ Trainer """
            ## paddle 貌似是在 train 函数中进行定义的，提前也可以吗？

            # self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
            ##self.G_optim = paddle.fluid.optimizer.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.999,
            ##                                                    epsilon=self.weight_decay, parameter_list=(self.genA2B.parameters()+self.genB2A.parameters()),
            ##                                                    regularization=None, grad_clip=None, name=None,
            ##                                                    lazy_mode=False)
            # self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
            ##self.D_optim = paddle.fluid.optimizer.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.999,
            ##                                                    epsilon=self.weight_decay, parameter_list=(self.disGA.parameters()+self.disGB.parameters()+self.disLA.parameters()+self.disLB.parameters()),
            ##                                                    regularization=None, grad_clip=None, name=None,
            ##                                                    lazy_mode=False)

            self.G_optim = paddle.fluid.optimizer.AdamOptimizer(
                learning_rate=fluid.dygraph.PolynomialDecay(
                    learning_rate = self.lr,
                    decay_steps = 45000,
                    end_learning_rate=0.000001,
                    power=1.0),
                beta1=0.5,
                beta2=0.999,
                parameter_list=(self.genA2B.parameters() + self.genB2A.parameters()),
                regularization=paddle.fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay))

            self.D_optim = paddle.fluid.optimizer.AdamOptimizer(
                learning_rate=fluid.dygraph.PolynomialDecay(
                    learning_rate = self.lr,
                    decay_steps = 45000,
                    end_learning_rate=0.000001,
                    power=1.0),
                beta1=0.5,
                beta2=0.999,
                parameter_list=(self.disGA.parameters() + self.disGB.parameters() + self.disLA.parameters() + self.disLB.parameters()),
                regularization=paddle.fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay))
            """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
            ## paddle 貌似是在 train 过程中进行实现的：获取参数+裁剪+回传参数

            self.Rho_clipper = RhoClipper(0, 1)

    def train(self):
        place = fluid.CUDAPlace(0) if self.device == 'cuda' else fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            # 把 networks 中的模型进行实例化
            # 原来放在了这里，但是后面梯度更新有问题，还是放回到 build_model函数中
            #self.genA2B = ResnetGenerator('genA2B',input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res,
            #                              img_size=self.img_size, light=self.light)
            #self.genB2A = ResnetGenerator('genB2A',input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res,
            #                              img_size=self.img_size, light=self.light)
            #self.disGA = Discriminator_G('disGA',input_nc=3, ndf=self.ch, n_layers=7)
            #self.disGB = Discriminator_G('disGB',input_nc=3, ndf=self.ch, n_layers=7)
            #self.disLA = Discriminator_L('disLA',input_nc=3, ndf=self.ch, n_layers=5)
            #self.disLB = Discriminator_L('disLB',input_nc=3, ndf=self.ch, n_layers=5)

            # 设定为训练模式
            self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

            start_iter = 1  # 初始化，表示从第一步开始
            # 如果是继续训练，则要读取前面存储的模型
            if self.resume:
                #model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pdparams'))
                model_list = glob(os.path.join(self.result_dir, '*.pdparams'))
                if not len(model_list) == 0:
                    model_list.sort()
                    start_iter = int(model_list[ -1 ].split('_')[ -1 ].split('.')[ 0 ])  # 更新到前面训练的步数
                    #self.load(os.path.join(self.result_dir, self.dataset, 'model'), start_iter)
                    self.load(self.result_dir, start_iter)
                    print(" [*] Load SUCCESS")
                    ##if self.decay_flag and start_iter > (self.iteration // 2):
                        # 这里重新载入模型后需要重新设置学习率
                        # self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                        # self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                        ##self.lr -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                        ##self.G_optim = paddle.fluid.optimizer.AdamOptimizer(learning_rate=self.lr, beta1=0.5,
                        ##                                                    beta2=0.999, epsilon=self.weight_decay,
                        ##                                                    parameter_list=None, regularization=None,
                        ##                                                    grad_clip=None, name=None, lazy_mode=False)
                        ##self.D_optim = paddle.fluid.optimizer.AdamOptimizer(learning_rate=self.lr, beta1=0.5,
                        ##                                                    beta2=0.999, epsilon=self.weight_decay,
                        ##                                                    parameter_list=None, regularization=None,
                        ##                                                    grad_clip=None, name=None, lazy_mode=False)

            # training loop
            print('training start !')
            start_time = time.time()
            for step in range(start_iter, self.iteration + 1):
                #print('step = ', step)
                ##if self.decay_flag and step > (self.iteration // 2):
                    # self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                    # self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                    ##self.lr -= (self.lr / (self.iteration // 2))
                    ##self.G_optim = paddle.fluid.optimizer.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.999,
                    ##                                                    epsilon=self.weight_decay, parameter_list=None,
                    ##                                                    regularization=None, grad_clip=None, name=None,
                    ##                                                    lazy_mode=False)
                    ##self.D_optim = paddle.fluid.optimizer.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.999,
                    ##                                                    epsilon=self.weight_decay, parameter_list=None,
                    ##                                                    regularization=None, grad_clip=None, name=None,
                    ##                                                    lazy_mode=False)

                #try:
                #    real_A, _ = trainA_iter.next()
                #except:
                #    trainA_iter = iter(self.trainA_loader)
                #    real_A, _ = trainA_iter.next()

                #try:
                #    real_B, _ = trainB_iter.next()
                #except:
                #    trainB_iter = iter(self.trainB_loader)
                #    real_B, _ = trainB_iter.next()
                try:
                    real_A = next(trainA_iter)
                except:
                    trainA_iter = iter(self.trainA_loader())
                    real_A = next(trainA_iter)

                try:
                    real_B = next(trainB_iter)
                except:
                    trainB_iter = iter(self.trainB_loader())
                    real_B = next(trainB_iter)

                # 这里设定运行place应该就不用了
                # real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                # Update D
                # 清除梯度
                # self.D_optim.zero_grad()
                self.D_optim.clear_gradients()


                # 在前向传播以前，先要转成 paddle 的tensor
                real_A = np.array(real_A).astype(dtype='float32')
                real_B = np.array(real_B).astype(dtype='float32')
                real_A = paddle.fluid.dygraph.to_variable(real_A)
                real_B = paddle.fluid.dygraph.to_variable(real_B)

                # 前向传输的过程貌似不用变
                fake_A2B, _, _ = self.genA2B(real_A)
                fake_B2A, _, _ = self.genB2A(real_B)

                real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
                real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
                real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
                real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

                # D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device))\
                #    + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
                D_ad_loss_GA = self.MSE_loss(real_GA_logit, paddle.fluid.layers.ones_like(real_GA_logit)) \
                               + self.MSE_loss(fake_GA_logit, paddle.fluid.layers.zeros_like(fake_GA_logit))

                # D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device))\
                #    + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
                D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, paddle.fluid.layers.ones_like(real_GA_cam_logit)) \
                                   + self.MSE_loss(fake_GA_cam_logit, paddle.fluid.layers.zeros_like(fake_GA_cam_logit))

                # D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device))\
                #    + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
                D_ad_loss_LA = self.MSE_loss(real_LA_logit, paddle.fluid.layers.ones_like(real_LA_logit)) \
                               + self.MSE_loss(fake_LA_logit, paddle.fluid.layers.zeros_like(fake_LA_logit))

                # D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device))\
                #    + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
                D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, paddle.fluid.layers.ones_like(real_LA_cam_logit)) \
                                   + self.MSE_loss(fake_LA_cam_logit, paddle.fluid.layers.zeros_like(fake_LA_cam_logit))

                # D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device))\
                #    + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
                D_ad_loss_GB = self.MSE_loss(real_GB_logit, paddle.fluid.layers.ones_like(real_GB_logit)) \
                               + self.MSE_loss(fake_GB_logit, paddle.fluid.layers.zeros_like(fake_GB_logit))

                # D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device))\
                #    + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
                D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, paddle.fluid.layers.ones_like(real_GB_cam_logit)) \
                                   + self.MSE_loss(fake_GB_cam_logit, paddle.fluid.layers.zeros_like(fake_GB_cam_logit))

                # D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device))\
                #    + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
                D_ad_loss_LB = self.MSE_loss(real_LB_logit, paddle.fluid.layers.ones_like(real_LB_logit)) \
                               + self.MSE_loss(fake_LB_logit, paddle.fluid.layers.zeros_like(fake_LB_logit))

                # D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.device))\
                #    + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))
                D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, paddle.fluid.layers.ones_like(real_LB_cam_logit)) \
                                   + self.MSE_loss(fake_LB_cam_logit, paddle.fluid.layers.zeros_like(fake_LB_cam_logit))

                D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
                D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

                Discriminator_loss = D_loss_A + D_loss_B

                # 反向传播应该一样的
                Discriminator_loss.backward()
                # self.D_optim.step()
                self.D_optim.minimize(Discriminator_loss)

                # Update G
                # self.G_optim.zero_grad()
                self.G_optim.clear_gradients()

                # 前向传播不变
                fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
                fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

                fake_A2B2A, _, _ = self.genB2A(fake_A2B)
                fake_B2A2B, _, _ = self.genA2B(fake_B2A)

                fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
                fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

                # 计算 loss
                # G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
                G_ad_loss_GA = self.MSE_loss(fake_GA_logit, paddle.fluid.layers.ones_like(fake_GA_logit))

                # G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
                G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, paddle.fluid.layers.ones_like(fake_GA_cam_logit))

                # G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
                G_ad_loss_LA = self.MSE_loss(fake_LA_logit, paddle.fluid.layers.ones_like(fake_LA_logit))

                # G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
                G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, paddle.fluid.layers.ones_like(fake_LA_cam_logit))

                # G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
                G_ad_loss_GB = self.MSE_loss(fake_GB_logit, paddle.fluid.layers.ones_like(fake_GB_logit))

                # G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
                G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, paddle.fluid.layers.ones_like(fake_GB_cam_logit))

                # G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
                G_ad_loss_LB = self.MSE_loss(fake_LB_logit, paddle.fluid.layers.ones_like(fake_LB_logit))

                # G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))
                G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, paddle.fluid.layers.ones_like(fake_LB_cam_logit))

                # 这里用 L1_loss 应该一样吧
                G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
                G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

                G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
                G_identity_loss_B = self.L1_loss(fake_B2B, real_B)
                #print(fake_B2A_cam_logit)
                # G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device))\
                #    + self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
                G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, paddle.fluid.layers.ones_like(fake_B2A_cam_logit)) \
                               + self.BCE_loss(fake_A2A_cam_logit, paddle.fluid.layers.zeros_like(fake_A2A_cam_logit))

                # G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device))\
                #    + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))
                G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit,
                                             paddle.fluid.layers.ones_like(fake_A2B_cam_logit)) \
                               + self.BCE_loss(fake_B2B_cam_logit,
                                               paddle.fluid.layers.zeros_like(fake_B2B_cam_logit, out=None))

                G_loss_A = self.adv_weight * (
                            G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
                G_loss_B = self.adv_weight * (
                            G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

                Generator_loss = G_loss_A + G_loss_B
                Generator_loss.backward()

                # self.G_optim.step()
                self.G_optim.minimize(Generator_loss)


                # clip parameter of AdaILN and ILN, applied after optimizer step
                # 截断函数已经修改，但是不知道对不对
                #self.genA2B.apply(self.Rho_clipper)
                #self.genB2A.apply(self.Rho_clipper)
                self.Rho_clipper(self.genA2B)
                self.Rho_clipper(self.genB2A)

                print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (
                step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))
                if step % self.print_freq == 0:
                    train_sample_num = 5
                    test_sample_num = 5
                    A2B = np.zeros((self.img_size * 7, 0, 3))
                    B2A = np.zeros((self.img_size * 7, 0, 3))

                    # 切换到测试模式，这里貌似不用改
                    self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                    for _ in range(train_sample_num):
                        #try:
                        #    real_A, _ = trainA_iter.next()
                        #except:
                        #    trainA_iter = iter(self.trainA_loader)
                        #    real_A, _ = trainA_iter.next()

                        #try:
                        #    real_B, _ = trainB_iter.next()
                        #except:
                        #    trainB_iter = iter(self.trainB_loader)
                        #    real_B, _ = trainB_iter.next()


                        try:
                            real_A = next(trainA_iter)
                        except:
                            trainA_iter = iter(self.trainA_loader())
                            real_A = next(trainA_iter)

                        try:
                            real_B = next(trainB_iter)
                        except:
                            trainB_iter = iter(self.trainB_loader())
                            real_B = next(trainB_iter)

                        # 应该不用再次设定 place 了
                        # real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                        # 在前向传播以前，先要转成 paddle 的tensor
                        real_A = np.array(real_A).astype(dtype='float32')
                        real_B = np.array(real_B).astype(dtype='float32')
                        real_A = fluid.dygraph.to_variable(real_A)
                        real_B = fluid.dygraph.to_variable(real_B)

                        # 前向传播不用改
                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[ 0 ]))),
                                                                   cam(tensor2numpy(fake_A2A_heatmap[ 0 ]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2A[ 0 ]))),
                                                                   cam(tensor2numpy(fake_A2B_heatmap[ 0 ]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B[ 0 ]))),
                                                                   cam(tensor2numpy(fake_A2B2A_heatmap[ 0 ]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B2A[ 0 ])))), 0)),
                                             1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[ 0 ]))),
                                                                   cam(tensor2numpy(fake_B2B_heatmap[ 0 ]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2B[ 0 ]))),
                                                                   cam(tensor2numpy(fake_B2A_heatmap[ 0 ]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A[ 0 ]))),
                                                                   cam(tensor2numpy(fake_B2A2B_heatmap[ 0 ]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A2B[ 0 ])))), 0)),
                                             1)

                    for _ in range(test_sample_num):
                        try:
                            real_A = next(testA_iter)
                        except:
                            testA_iter = iter(self.testA_loader())
                            real_A = next(testA_iter)

                        try:
                            real_B = next(testB_iter)
                        except:
                            testB_iter = iter(self.testB_loader())
                            real_B = next(testB_iter)
                        #real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                        # 在前向传播以前，先要转成 paddle 的tensor
                        real_A = np.array(real_A).astype(dtype='float32')
                        real_B = np.array(real_B).astype(dtype='float32')
                        real_A = paddle.fluid.dygraph.to_variable(real_A)
                        real_B = paddle.fluid.dygraph.to_variable(real_B)

                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[ 0 ]))),
                                                                   cam(tensor2numpy(fake_A2A_heatmap[ 0 ]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2A[ 0 ]))),
                                                                   cam(tensor2numpy(fake_A2B_heatmap[ 0 ]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B[ 0 ]))),
                                                                   cam(tensor2numpy(fake_A2B2A_heatmap[ 0 ]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B2A[ 0 ])))), 0)),
                                             1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[ 0 ]))),
                                                                   cam(tensor2numpy(fake_B2B_heatmap[ 0 ]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2B[ 0 ]))),
                                                                   cam(tensor2numpy(fake_B2A_heatmap[ 0 ]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A[ 0 ]))),
                                                                   cam(tensor2numpy(fake_B2A2B_heatmap[ 0 ]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A2B[ 0 ])))), 0)),
                                             1)

                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                    self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

                if step % self.save_freq == 0:
                    self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

                if step % 5000 == 0:
                    #params = {}
                    #params[ 'genA2B' ] = self.genA2B.state_dict()
                    #params[ 'genB2A' ] = self.genB2A.state_dict()
                    #params[ 'disGA' ] = self.disGA.state_dict()
                    #params[ 'disGB' ] = self.disGB.state_dict()
                    #params[ 'disLA' ] = self.disLA.state_dict()
                    #params[ 'disLB' ] = self.disLB.state_dict()
                    # torch.save(params, os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))
                    paddle.fluid.dygraph.save_dygraph(self.genA2B.state_dict(),
                                                      os.path.join(self.result_dir,
                                                                   self.dataset + '_genA2B_params_%07d' % step))
                    paddle.fluid.dygraph.save_dygraph(self.genB2A.state_dict(),
                                                      os.path.join(self.result_dir,
                                                                   self.dataset + '_genB2A_params_%07d' % step))
                    paddle.fluid.dygraph.save_dygraph(self.disGA.state_dict(),
                                                      os.path.join(self.result_dir,
                                                                   self.dataset + '_disGA_params_%07d' % step))
                    paddle.fluid.dygraph.save_dygraph(self.disGB.state_dict(),
                                                      os.path.join(self.result_dir,
                                                                   self.dataset + '_disGB_params_%07d' % step))
                    paddle.fluid.dygraph.save_dygraph(self.disLA.state_dict(),
                                                      os.path.join(self.result_dir,
                                                                   self.dataset + '_disLA_params_%07d' % step))
                    paddle.fluid.dygraph.save_dygraph(self.disLB.state_dict(),
                                                      os.path.join(self.result_dir,
                                                                   self.dataset + '_disLB_params_%07d' % step))
                    paddle.fluid.dygraph.save_dygraph(self.G_optim.state_dict(),
                                                      os.path.join(self.result_dir,
                                                                   self.dataset + '_genA2B_params_%07d' % step))
                    paddle.fluid.dygraph.save_dygraph(self.D_optim.state_dict(),
                                                      os.path.join(self.result_dir,
                                                                   self.dataset + '_disGA_params_%07d' % step))

    def save(self, dir, step):
        #params = {}
        #params[ 'genA2B' ] = self.genA2B.state_dict()
        #params[ 'genB2A' ] = self.genB2A.state_dict()
        #params[ 'disGA' ] = self.disGA.state_dict()
        #params[ 'disGB' ] = self.disGB.state_dict()
        #params[ 'disLA' ] = self.disLA.state_dict()
        #params[ 'disLB' ] = self.disLB.state_dict()
        # torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        paddle.fluid.dygraph.save_dygraph(self.genA2B.state_dict(),
                                          os.path.join(dir, self.dataset + '_genA2B_params_%07d' % step))
        paddle.fluid.dygraph.save_dygraph(self.genB2A.state_dict(),
                                          os.path.join(dir, self.dataset + '_genB2A_params_%07d' % step))
        paddle.fluid.dygraph.save_dygraph(self.disGA.state_dict(),
                                          os.path.join(dir, self.dataset + '_disGA_params_%07d' % step))
        paddle.fluid.dygraph.save_dygraph(self.disGB.state_dict(),
                                          os.path.join(dir, self.dataset + '_disGB_params_%07d' % step))
        paddle.fluid.dygraph.save_dygraph(self.disLA.state_dict(),
                                          os.path.join(dir, self.dataset + '_disLA_params_%07d' % step))
        paddle.fluid.dygraph.save_dygraph(self.disLB.state_dict(),
                                          os.path.join(dir, self.dataset + '_disLB_params_%07d' % step))
        paddle.fluid.dygraph.save_dygraph(self.G_optim.state_dict(),
                                          os.path.join(dir, self.dataset + '_genA2B_params_%07d' % step))
        paddle.fluid.dygraph.save_dygraph(self.D_optim.state_dict(),
                                          os.path.join(dir, self.dataset + '_disGA_params_%07d' % step))

    def load(self, dir, step):
        # params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        #self.genA2B.load_state_dict(params[ 'genA2B' ])
        #self.genB2A.load_state_dict(params[ 'genB2A' ])
        #self.disGA.load_state_dict(params[ 'disGA' ])
        #self.disGB.load_state_dict(params[ 'disGB' ])
        #self.disLA.load_state_dict(params[ 'disLA' ])
        #self.disLB.load_state_dict(params[ 'disLB' ])
        genA2B_params, G_optim_params = paddle.fluid.dygraph.load_dygraph(
            os.path.join(dir, self.dataset + '_genA2B_params_%07d' % step))
        self.genA2B.set_dict(genA2B_params)
        # 不要优化器参数试试
        #self.G_optim.set_dict(G_optim_params)
        genB2A_params, _ = paddle.fluid.dygraph.load_dygraph(os.path.join(dir, self.dataset + '_genB2A_params_%07d' % step))
        self.genB2A.set_dict(genB2A_params)
        disGA_params, D_optim_params = paddle.fluid.dygraph.load_dygraph(
            os.path.join(dir, self.dataset + '_disGA_params_%07d' % step))
        self.disGA.set_dict(disGA_params)
        # 不要优化器参数试试
        #self.D_optim.set_dict(D_optim_params)
        disGB_params, _ = paddle.fluid.dygraph.load_dygraph(os.path.join(dir, self.dataset + '_disGB_params_%07d' % step))
        self.disGB.set_dict(disGB_params)
        disLA_params, _ = paddle.fluid.dygraph.load_dygraph(os.path.join(dir, self.dataset + '_disLA_params_%07d' % step))
        self.disLA.set_dict(disLA_params)
        disLB_params, _ = paddle.fluid.dygraph.load_dygraph(os.path.join(dir, self.dataset + '_disLB_params_%07d' % step))
        self.disLB.set_dict(disLB_params)
        #G_optim_params = paddle.fluid.dygraph.load_dygraph(
        #    os.path.join(dir, self.dataset + '_genA2B_params_%07d' % step))
        #self.G_optim.set_dict(G_optim_params)
        #D_optim_params = paddle.fluid.dygraph.load_dygraph(
        #    os.path.join(dir, self.dataset + '_disGA_params_%07d' % step))
        #self.D_optim.set_dict(D_optim_params)


    def test(self):
        # model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
        #model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pdparams'))
        #if not len(model_list) == 0:
        #    model_list.sort()
        #    iter = int(model_list[ -1 ].split('_')[ -1 ].split('.')[ 0 ])
        #    self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
        #    print(" [*] Load SUCCESS")
        #else:
        #    print(" [*] Load FAILURE")
        #    return

        place = fluid.CUDAPlace(0) if self.device == 'cuda' else fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            model_list = glob(os.path.join(self.result_dir, '*.pdparams'))
            if not len(model_list) == 0:
                model_list.sort()
                iter = int(model_list[ -1 ].split('_')[ -1 ].split('.')[ 0 ])
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
                print(" [*] Load SUCCESS")
            else:
                print(" [*] Load FAILURE")
                return
            self.genA2B.eval(), self.genB2A.eval()
            for n, real_A in enumerate(self.testA_loader()):
                # 无需设定 place
                # real_A = real_A.to(self.device)
    
                # 在前向传播以前，先要转成 paddle 的tensor
                real_A = np.array(real_A).astype(dtype='float32')
                real_A = paddle.fluid.dygraph.to_variable(real_A)
    
                # 前向传播不改
                fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
    
                fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
    
                fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
    
                #A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[ 0 ]))),
                #                      cam(tensor2numpy(fake_A2A_heatmap[ 0 ]), self.img_size),
                #                      RGB2BGR(tensor2numpy(denorm(fake_A2A[ 0 ]))),
                #                      cam(tensor2numpy(fake_A2B_heatmap[ 0 ]), self.img_size),
                #                      RGB2BGR(tensor2numpy(denorm(fake_A2B[ 0 ]))),
                #                      cam(tensor2numpy(fake_A2B2A_heatmap[ 0 ]), self.img_size),
                #                      RGB2BGR(tensor2numpy(denorm(fake_A2B2A[ 0 ])))), 0)
                A2B = RGB2BGR(tensor2numpy(denorm(fake_A2B[ 0 ])))
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)
    
            for n, real_B in enumerate(self.testB_loader()):
                # 无需设定 place
                # real_B = real_B.to(self.device)
    
                # 在前向传播以前，先要转成 paddle 的tensor
                real_B = np.array(real_B).astype(dtype='float32')
                real_B = paddle.fluid.dygraph.to_variable(real_B)
    
                # 前向传播不改
                fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)
    
                fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)
    
                fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)
    
                #B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[ 0 ]))),
                #                      cam(tensor2numpy(fake_B2B_heatmap[ 0 ]), self.img_size),
                #                      RGB2BGR(tensor2numpy(denorm(fake_B2B[ 0 ]))),
                #                      cam(tensor2numpy(fake_B2A_heatmap[ 0 ]), self.img_size),
                #                      RGB2BGR(tensor2numpy(denorm(fake_B2A[ 0 ]))),
                #                      cam(tensor2numpy(fake_B2A2B_heatmap[ 0 ]), self.img_size),
                #                      RGB2BGR(tensor2numpy(denorm(fake_B2A2B[ 0 ])))), 0)
                B2A = RGB2BGR(tensor2numpy(denorm(fake_B2A[ 0 ])))
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)
