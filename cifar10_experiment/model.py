class InitialP4Conv(nn.Module):
  def __init__(self):
      super(InitialP4Conv,self).__init__()
      self.conv = nn.Sequential(
                                P4ConvZ2(1,32,5),
                                nn.BatchNorm3d(32),
                                nn.ReLU(),
                                P4ConvP4(32,32,3),
                                nn.BatchNorm3d(32),
                                nn.ReLU(), 
                                P4ConvP4(32,32,5),
                                nn.BatchNorm3d(32),
                                nn.ReLU()   
                               )
  def forward(self,x):
      out = self.conv(x)     
      return out                   

class PrimaryCapsules(nn.Module):
  '''Use 2d convolution to extract capsules'''
  def __init__(self,num_capsules=10,in_channels=32,out_channels=32):
    super(PrimaryCapsules, self).__init__()
    self.num_capsules = num_capsules
    self.out_channels = out_channels
    self.capsules = nn.Sequential(
                                  P4ConvP4(in_channels,out_channels*num_capsules,3),
                                  nn.BatchNorm3d(out_channels*num_capsules),
                                  nn.ReLU(),
                                 )   
       
  def squash(self,x,dim):
    norm_squared = (x ** 2).sum(dim, keepdim=True)
    part1 = norm_squared / (1 +  norm_squared)
    part2 = x / torch.sqrt(norm_squared+ 1e-16)
    output = part1 * part2 
    return output
  
  def forward(self, x):
    output = self.capsules(x)
    H, W = output.size(-2), output.size(-1)
    output = output.view(-1,self.num_capsules,self.out_channels,4,H,W)
    output = self.squash(output,dim=2)    
    return output

class ConvolutionalCapsules(nn.Module):
  '''
      A capsule layer that uses one convolution per capsule-type
  '''
  def __init__(self,num_in_capsules,in_capsule_dim,num_out_capsules,out_capsule_dim,kernel_size,stride=1,dilation=1,padding=0):
    super(ConvolutionalCapsules,self).__init__()
    self.num_in_capsules = num_in_capsules
    self.in_capsule_dim = in_capsule_dim
    self.num_out_capsules = num_out_capsules
    self.out_capsule_dim = out_capsule_dim
    self.kernel_size = kernel_size
    self.stride = stride
    self.dilation = dilation
    self.padding = padding
    self.projection_network = nn.Sequential(
                                            P4ConvP4(in_capsule_dim,out_capsule_dim*num_out_capsules,kernel_size,stride,padding,dilation),
                                            nn.BatchNorm3d(out_capsule_dim*num_out_capsules)
                                           )  
     
  def squash(self,x,dim):
    norm_squared = (x ** 2).sum(dim, keepdim=True)
    part1 = norm_squared / (1 +  norm_squared)
    part2 = x / torch.sqrt(norm_squared+ 1e-16)
    output = part1 * part2 
    return output
   
  def cosine_similarity(self,predictions,eps=1e-8):
    dot_product = torch.matmul(predictions,predictions.transpose(-1,-2))
    norm_sq = torch.norm(predictions,dim=-1,keepdim=True)**2 
    eps_matrix = eps*torch.ones_like(norm_sq)
    norm_sq = torch.max(norm_sq,eps_matrix)
    similarity_matrix = dot_product/norm_sq
    return similarity_matrix    

  def degree_routing(self,capsules):
    batch_size = capsules.size(0)
    grid_size = [capsules.size(-3),capsules.size(-2),capsules.size(-1)]
    ### Prediction u_hat ####
    capsules = capsules.view(batch_size*self.num_in_capsules,self.in_capsule_dim,grid_size[0],grid_size[1],grid_size[2])                 
    u_hat = self.projection_network(capsules)
    grid_size = [u_hat.size(-3),u_hat.size(-2),u_hat.size(-1)]
    u_hat = u_hat.view(batch_size,self.num_in_capsules,self.num_out_capsules,self.out_capsule_dim,grid_size[0],grid_size[1],grid_size[2])   
    ### u_hat:(batch_size,num_in_capsules,num_out_capsules,out_capsule_dim,4,H,W)
    u_hat_permute = u_hat.permute(0,2,4,5,6,1,3)#(batch_size,num_out_capsules,4,H,W,num_in_capsules,out_capsule_dim)
    affinity_matrices = self.cosine_similarity(u_hat_permute)    
    degree_score = F.softmax(torch.sum(affinity_matrices,dim=-1,keepdim=True),dim=5)#(batch_size,num_out_capsules,4,H,W,num_in_capsules,1)
    degree_score = (degree_score).permute(0,5,1,6,2,3,4)#(batch_size,num_in_capsules,num_out_capsules,1,4,H,W)
    s_j = (degree_score * u_hat).sum(dim=1)
    v_j = self.squash(s_j,dim=3)
    return v_j.squeeze(dim=1), degree_score.squeeze(dim=3) 

  def forward(self,capsules,routing_iterations=1):
    '''
        Input: (batch_size, num_in_capsules, in_capsule_dim, H, W)
        Output: (batch_size, num_out_capsules, out_capsule_dim, H', W')
    '''
    out_capsules, degree_score = self.degree_routing(capsules)
    return out_capsules, degree_score

class CapsuleDimensionChange(nn.Module):
      super(CapsuleDimensionChange, self).__init__()
      def __init__(num_in_capsules, in_capsule_dim, num_out_capsules, out_capsule_dim):
          self.conv_1 = P4ConvP4(num_in_capsule*in_capsule_dim, num_out_capsules*out_capsule_dim, kernel_size=1)
          self.num_out_capsules = num_out_capsules
          self.out_capsule_dim = out_capsule_dim
      
      def forward(self, in_capsule):
          _, num_in_capsule, in_capsule_dim, _, H, W
          in_capsule = in_capsule.view(-1, num_in_capsule*in_capsule_dim, 4, H, W)
          out_capsule = self.conv_1(in_capsule)
          out_capsule = out_capsule.view(-1, self.num_out_capsules, self.out_capsule_dim, 4, H, W)
          return out_capsule

class BasicSovNetBlock(nn.Module):
  '''A residual SOVNET block'''
  def __init__(self,num_in_capsules, in_capsule_dim, num_out_capsules, out_capsule_dim, stride=1, dilation=1, padding=0):
    super(ResidualSovBlock,self).__init__()
    self.conv_sov1 = ConvolutionalCapsules(num_in_capsules, in_capsule_dim, num_out_capsules, out_capsule_dim, kernel_size=3, stride=1, padding=0, dilation=1)
    self.conv_sov2 = ConvolutionalCapsules(num_out_capsules*2, out_capsule_dim, num_out_capsules, out_capsule_dim, kernel_size=3, stride=1, padding=padding, dilation=dilation)
    self.conv_change_dim = CapsulesDimensionChange(num_in_capsules, in_capsule_dim, num_out_capsules, out_capsule_dim)
    self.shortcut = nn.Sequential()
    if dilation!=1 or stride != 1 or in_capsule_dim != out_capsule_dim:
            self.shortcut = nn.Sequential(
                                          ConvolutionalCapsules(num_out_capsules, out_capsule_dim, num_out_capsules, out_capsule_dim, kernel_size=1, stride=stride, padding=0, dilation=dilation),
            )
  
  def squash(self,x,dim):
    norm_squared = (x ** 2).sum(dim, keepdim=True)
    part1 = norm_squared / (1 +  norm_squared)
    part2 = x / torch.sqrt(norm_squared+ 1e-16)
    output = part1 * part2 
    return output
  
  def forward(self,in_capsule):
    '''
        Input: (batch_size,num_in_capsules,in_capsule_dim,H,W)
        Output: (batch_size,num_out_capsules,out_capsule_dim,H',W')
    '''
    out_capsule = self.conv_sov1(in_capsule)
    #out_capsule = self.conv_sov2(out_capsule)
    #out_capsule = self.conv_sov3(out_capsule)
    in_capsule = self.conv_change_dim(in_capsule)
    in_capsule = self.shortcut(in_capsule)
    out_capsule = torch.cat((in_capsule, out_capsule), 0).contiguous()
    out_capsule = self.conv_sov2(out_capsule)
    #out_capsule += self.shortcut(in_capsule)
    #out_capsule = self.squash(out_capsule,dim=2)
    return out_capsule

class ResidualSovnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResidualSovnet, self).__init__()
        self.in_capsule_dim = 4
        self.num_capsules = 4
        self.conv1 = InitialP4Conv()
        self.primary_capsules = PrimaryCapsules(self.num_capsules,32,4)
        self.layer1 = self._make_layer(block, 4, 4, 6, 4, num_blocks[0], stride=1, padding=0, dilation=1)
        self.layer2 = self._make_layer(block, 6, 4, 8, 8, num_blocks[1], stride=2, padding=1, dilation=2)
        self.layer3 = self._make_layer(block, 8, 8, 10, 16, num_blocks[2], stride=2, padding=1, dilation=2)
        self.layer4 = self._make_layer(block, 10, 16, 16, 16, num_blocks[3], stride=2, padding=0, dilation=4)
        self.class_capsules = ConvolutionalCapsules(self.num_capsules,16,16,10,16,5)
        self.reconstruction_layer = ReconstructionLayer(10,16,31,3).to(device)
        
    def _make_layer(self, block, num_in_capsule, in_capsule_dim, num_out_capsule, out_capsule_dim, num_blocks, stride=1, padding=0, dilation=1):
        #strides = [stride] + [1]*(num_blocks-1)
        dilations = [1]*(num_blocks-1)
        layers = []
        layers.append(block(num_in_capsule, in_capsule_dim, num_out_capsule, out_capsule_dim, 1, padding=0, dilation=dilation))
        for dilation in dilations:
            layers.append(block(num_out_capsule, out_capsule_dim, num_out_capsule, out_capsule_dim, 1, padding=1, dilation=1))
        #self.in_capsule_dim = out_capsule_dim
        #self.num_capsules = 2*self.num_capsules
        return nn.Sequential(*layers)

    def forward(self, x, target):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.primary_capsules(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        class_capsules = self.class_capsules(out)
        reconstructions, masked = self.reconstruction_layer(class_capsules,target)
        return class_capsules, reconstructions, masked
  
    def get_activations(self,capsules):
      return torch.norm(capsules, dim=2).squeeze()
       
    def get_predictions(self,activations):
      max_length_indices = activations.max(dim=1)[1].squeeze()#(batch_size)
      predictions = activations.new_tensor(torch.eye(10))
      predictions = predictions.index_select(dim=0,index=max_length_indices)
      return predictions
