from parlai.scripts.eval_model import eval_model, setup_args as base_setup_args

IS_ORIGINAL = True


def setup_task():
    if IS_ORIGINAL:
        task_name = 'tasks.convai2transmitter.agents:BothOriginalTeacher'
    else:
        task_name = 'tasks.convai2transmitter.agents:BothRevisedTeacher'
    return task_name


def setup_trained_weights():
    if IS_ORIGINAL:
        weights_name = 'tmp/transmitter/selforiginal_dialogpt_gpt2dict/selforiginal_dialogpt_gpt2dict.model'
    else:
        weights_name = 'tmp/transmitter/bothrevised_gpt2/bothrevised_gpt2.model'
    return weights_name


def setup_args(parser=None):
    parser = base_setup_args(parser)
    task_name = setup_task()
    parser.set_defaults(
        task=task_name,
        datatype='valid',
        hide_labels=False,
        metrics='apcer,bpcer',
    )
    return parser


def eval_hits(opt, print_parser):
    report = eval_model(opt, print_parser)
    print('============================')
    print('FINAL Hits@1: ' + str(report['hits@1']))


if __name__ == '__main__':
    parser = setup_args()
    model_name = setup_trained_weights()
    parser.set_params(
        model='agents.transmitter.transmitter:TransformerAgent',
        model_file=model_name,
        init_model_transmitter=model_name,
        gpt_type='dialogpt',
        gpu=1,
        batchsize=10,
        beam_size=1,
        rank_candidates=True,
        report_freq=0.0001,
    )
    print("Model: {}".format(model_name))
    opt = parser.parse_args(print_args=False)
    eval_hits(opt, print_parser=parser)