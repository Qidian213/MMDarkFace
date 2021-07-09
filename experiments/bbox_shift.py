@PIPELINES.register_module()
class RandomBoxShift(object):
    def __init__(self, shift_ratio=None):
        self.shift_ratio = shift_ratio

    def bbox_shift(self, bboxes, img_shape, shift_ratio):
        assert bboxes.shape[-1] % 4 == 0
        shifted = bboxes.copy()
        
        s1 = (1-shift_ratio)/2.0
        s2 = (1+shift_ratio)/2.0
        shifted[..., 0::4] = s2*bboxes[..., 0::4] + s1*bboxes[..., 2::4]
        shifted[..., 2::4] = s1*bboxes[..., 0::4] + s2*bboxes[..., 2::4]
        shifted[..., 1::4] = s2*bboxes[..., 1::4] + s1*bboxes[..., 3::4]
        shifted[..., 3::4] = s1*bboxes[..., 1::4] + s2*bboxes[..., 3::4]

        shifted[:, 0::2] = np.clip(shifted[:, 0::2], 0, img_shape[1])
        shifted[:, 1::2] = np.clip(shifted[:, 1::2], 0, img_shape[0])
        
        return shifted

    def __call__(self, results):
        shift_ratio = random.uniform(self.shift_ratio[0], self.shift_ratio[1])
        
        if(shift_ratio != 1.0):
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_shift(results[key], results['img_shape'], shift_ratio)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(shift_ratio={self.shift_ratio})'
