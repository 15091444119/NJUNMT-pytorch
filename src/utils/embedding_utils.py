from src.utils.logging import INFO


def write_embeddings(embs, vocab, path):
    """ Write embs in to path
    Args:
        embs(numpy array):
        vocab(Vocabulary)
        path
    """
    INFO("Write embeddings into {}...".format(path))
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(str(embs.shape[0]) + ' ' + str(embs.shape[1]) + '\n')
        for idx in range(len(embs)):
            token = vocab.id2token(idx)
            f.writelines(token + ' ' + ' '.join([str(x) for x in embs[idx]]) + '\n')
    INFO("Finished.")


def get_embeddings(nmt_model, vocab_src, vocab_tgt, path_prefix):
    """ get embeddings from nmt_model and write to file
    Args:
        nmt_model: model of nmt ,dl4mt or transformer
        vocab_src(Vocabulary)
        vocab_tgt(Vocabulary)
        path_prefix: source and target embeddings will be saved in path_prefix + ".src.txt" and + "tgt.txt"
    """
    src_embeddings = nmt_model.encoder.embeddings.embeddings.weight.detach().cpu().numpy()
    tgt_embeddings = nmt_model.decoder.embeddings.embeddings.weight.detach().cpu().numpy()
    write_embeddings(src_embeddings, vocab_src, path_prefix + ".src.txt")
    write_embeddings(tgt_embeddings, vocab_tgt, path_prefix + ".tgt.txt")
    exit()
