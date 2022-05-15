#!/usr/bin/env bash

if [[ $PUBLIC_KEY ]]; then
	mkdir --parents ~/.ssh && chmod a+rwx,g-rwx,o-rwx ~/.ssh
	echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
	chmod a+rwx,u-wx,g-rwx,o-rwx ~/.ssh/authorized_keys
	service ssh start
fi

echo "Pod started"
sleep infinity
