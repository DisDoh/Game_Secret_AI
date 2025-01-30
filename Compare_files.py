def compare_binary_files(file1, file2):
    """
    Compares two binary files and returns their similarity percentage
    based on the proportion of identical bytes.

    :param file1: Path to the first binary file
    :param file2: Path to the second binary file
    :return: Similarity percentage (float)
    """
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        data1 = f1.read()
        data2 = f2.read()

    # Length of both files
    len1 = len(data1)
    len2 = len(data2)

    # Handle empty file cases
    if len1 == 0 or len2 == 0:
        # Decide how to handle empty files; for instance,
        # return 100 if both are empty, or 0 if only one is empty
        return 100.0 if (len1 == 0 and len2 == 0) else 0.0

    # Calculate the number of identical bytes in the common portion
    min_length = min(len1, len2)
    identical_bytes = sum(b1 == b2 for b1, b2 in zip(data1[:min_length], data2[:min_length]))

    # Define similarity as the number of identical bytes divided by the length
    # of the longer file, then multiply by 100 to get a percentage
    similarity = (identical_bytes / max(len1, len2)) * 100
    return similarity


# Example usage
if __name__ == "__main__":
    file_bin1 = "temp_container.tar.xz(4).AIZip"
    file_bin2 = "temp_container.tar.xz_.AIZip"

    similarity_score = compare_binary_files(file_bin1, file_bin2)
    print(f"The files have a similarity of {similarity_score:.2f}%")

    file_bin1 = "model(4).pkl"
    file_bin2 = "model.pkl"

    similarity_score = compare_binary_files(file_bin1, file_bin2)
    print(f"The files have a similarity of {similarity_score:.2f}%")
